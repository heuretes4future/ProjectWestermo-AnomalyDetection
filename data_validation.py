# data_validation.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# --- Helpers -----------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return -1


@dataclass
class ValidationConfig:
    # Expected rows from paper (reduced dataset)
    expected_rows: Dict[str, int]
    # Required columns
    required_cols: List[str]
    # Expected protocol categories (from your config)
    protocol_allowed: List[str]
    # Label columns
    binary_label: str
    multiclass_label: str
    # If strict: fail on any mismatch
    strict: bool = True


DEFAULT_VALIDATION_CONFIG = ValidationConfig(
    expected_rows={"Right": 4720, "Bottom": 35404, "Left": 8533},
    required_cols=["protocol", "NST_B_Label", "NST_M_Label"],
    protocol_allowed=["ARP", "IPV4-TCP", "IPV4-UDP", "IPV4-2", "LLDP", "IPV4-ICMP"],
    binary_label="NST_B_Label",
    multiclass_label="NST_M_Label",
    strict=True,
)


# --- Core validation ----------------------------------------------------------

def validate_single_file(
    node_name: str,
    file_path: str,
    cfg: ValidationConfig,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "node": node_name,
        "file_path": file_path,
        "exists": os.path.exists(file_path),
    }

    if not out["exists"]:
        out["error"] = "File not found"
        return out

    out["sha256"] = sha256_file(file_path)

    # Read CSV
    df = pd.read_csv(file_path)
    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])

    # Check expected rows (paper)
    exp = cfg.expected_rows.get(node_name)
    out["expected_rows"] = exp
    out["rows_match_expected"] = (exp == out["rows"]) if exp is not None else None

    # Required columns present?
    missing_required = [c for c in cfg.required_cols if c not in df.columns]
    out["missing_required_columns"] = missing_required

    # Protocol domain check (if protocol exists)
    if "protocol" in df.columns:
        prot_unique = sorted(df["protocol"].dropna().astype(str).unique().tolist())
        out["protocol_unique"] = prot_unique
        unknown = [p for p in prot_unique if p not in cfg.protocol_allowed]
        out["protocol_unknown_values"] = unknown
    else:
        out["protocol_unique"] = None
        out["protocol_unknown_values"] = None

    # Label distributions & consistency checks
    bcol = cfg.binary_label
    mcol = cfg.multiclass_label

    if bcol in df.columns:
        out["binary_label_counts"] = df[bcol].value_counts(dropna=False).to_dict()
    else:
        out["binary_label_counts"] = None

    if mcol in df.columns:
        out["multiclass_label_counts"] = df[mcol].value_counts(dropna=False).to_dict()
    else:
        out["multiclass_label_counts"] = None

    # Consistency: Normal should align with binary=0 (before preprocessing)
    # We allow two common formats:
    # - binary numeric 0/1
    # - binary strings like "Normal"/"Attack" (rare)
    consistency = {"checked": False}
    if bcol in df.columns and mcol in df.columns:
        consistency["checked"] = True

        m = df[mcol].astype(str)

        b = df[bcol]
        # Normalize binary to numeric where possible
        b_num = pd.to_numeric(b, errors="coerce")

        if b_num.notna().any():
            # Treat "Normal" in multiclass -> binary should be 0
            normal_mask = (m == "Normal") | (m == "GOOD-SSH") | (m == "Good-SSH")
            # If Good-SSH exists pre-merge, it may be labeled normal or anomaly depending on dataset labeling.
            # We'll just check Normal specifically here.
            normal_only = (m == "Normal")
            bad_normal = int(((b_num == 1) & normal_only).sum())
            good_anom = int(((b_num == 0) & (m != "Normal")).sum())

            consistency["normal_multiclass_but_binary_1"] = bad_normal
            consistency["non_normal_multiclass_but_binary_0"] = good_anom
        else:
            # Can't interpret as numeric, just record unique values
            consistency["binary_unique_values"] = sorted(b.astype(str).unique().tolist())

    out["label_consistency"] = consistency

    # Duplicates
    out["duplicate_rows"] = int(df.duplicated().sum())

    # Missing values (overall + per column top 10)
    missing_total = int(df.isna().sum().sum())
    out["missing_total"] = missing_total
    miss_by_col = df.isna().sum().sort_values(ascending=False)
    out["missing_top10_cols"] = {k: int(v) for k, v in miss_by_col.head(10).to_dict().items() if v > 0}

    # Constant columns (nunique == 1) excluding labels and obvious identifiers
    label_cols = [c for c in [bcol, mcol] if c in df.columns]
    candidate_cols = [c for c in df.columns if c not in label_cols]
    nunique = df[candidate_cols].nunique(dropna=False)
    constants = nunique[nunique <= 1].index.tolist()
    out["constant_feature_columns"] = constants

    # Basic numeric sanity: count negative values per numeric column (top 10)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    neg_counts = {}
    for c in num_cols:
        # Some cols can be offsets etc; we still log negatives for inspection
        neg = int((df[c] < 0).sum(skipna=True))
        if neg > 0:
            neg_counts[c] = neg
    # top 10 by count
    out["negative_value_cols_top10"] = dict(sorted(neg_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    return out


def validate_dataset(
    files: Dict[str, str],
    cfg: ValidationConfig,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "strict": cfg.strict,
        "nodes": {},
        "summary": {},
        "status": "OK",
        "errors": [],
        "warnings": [],
    }

    # Per node
    for node, path in files.items():
        node_rep = validate_single_file(node, path, cfg)
        report["nodes"][node] = node_rep

    # Summary checks
    total_rows = sum(safe_int(report["nodes"][n].get("rows")) for n in report["nodes"])
    report["summary"]["total_rows"] = total_rows

    expected_total = sum(cfg.expected_rows.values())
    report["summary"]["expected_total_rows"] = expected_total
    report["summary"]["total_rows_match_expected"] = (total_rows == expected_total)

    # Fail conditions
    for node, rep in report["nodes"].items():
        if not rep.get("exists"):
            report["errors"].append(f"{node}: missing file: {rep.get('file_path')}")
            continue

        if rep.get("missing_required_columns"):
            report["errors"].append(f"{node}: missing required columns: {rep['missing_required_columns']}")

        # Unknown protocol values should at least warn (or error if strict)
        unknown = rep.get("protocol_unknown_values") or []
        if unknown:
            msg = f"{node}: unknown protocol values: {unknown}"
            (report["errors"] if cfg.strict else report["warnings"]).append(msg)

        # Row count mismatch: error in strict
        if rep.get("rows_match_expected") is False:
            msg = f"{node}: row count mismatch. got {rep.get('rows')} expected {rep.get('expected_rows')}"
            (report["errors"] if cfg.strict else report["warnings"]).append(msg)

        # Duplicates: warn
        if rep.get("duplicate_rows", 0) > 0:
            report["warnings"].append(f"{node}: duplicated rows: {rep['duplicate_rows']}")

        # Label consistency: warn if suspicious numbers are non-zero
        cons = rep.get("label_consistency", {})
        if cons.get("checked") and isinstance(cons.get("normal_multiclass_but_binary_1"), int):
            if cons["normal_multiclass_but_binary_1"] > 0:
                report["warnings"].append(
                    f"{node}: {cons['normal_multiclass_but_binary_1']} rows are multiclass Normal but binary=1"
                )
        if cons.get("checked") and isinstance(cons.get("non_normal_multiclass_but_binary_0"), int):
            # This can be valid for GOOD-SSH etc; still a good indicator
            if cons["non_normal_multiclass_but_binary_0"] > 0:
                report["warnings"].append(
                    f"{node}: {cons['non_normal_multiclass_but_binary_0']} rows are multiclass != Normal but binary=0"
                )

    # Total mismatch: error if strict
    if report["summary"]["total_rows_match_expected"] is False:
        msg = f"Total rows mismatch. got {total_rows} expected {expected_total}"
        (report["errors"] if cfg.strict else report["warnings"]).append(msg)

    if report["errors"]:
        report["status"] = "FAILED"

    return report


# --- CLI ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Westermo CSV dataset (reduced vs extended, labels, schema).")
    parser.add_argument("--strict", action="store_true", help="Fail on mismatches (recommended).")
    parser.add_argument("--soft", action="store_true", help="Only warn on mismatches.")
    parser.add_argument("--out", default="runs/data_validation_report.json", help="Where to save JSON report.")
    parser.add_argument("--config", default="config_westermo.py", help="Path to config file (default: config_westermo.py).")
    args = parser.parse_args()

    # Import project config dynamically
    # This assumes config_westermo.py is importable from cwd
    from config_westermo import WESTERMO_CONFIG  # type: ignore

    strict = True
    if args.soft:
        strict = False
    if args.strict:
        strict = True

    cfg = DEFAULT_VALIDATION_CONFIG
    cfg.strict = strict

    files = WESTERMO_CONFIG["files"]

    report = validate_dataset(files, cfg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nValidation status: {report['status']}")
    print(f"Report saved to: {args.out}")

    if report["errors"]:
        print("\nERRORS:")
        for e in report["errors"]:
            print(f" - {e}")

    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print(f" - {w}")

    return 1 if report["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
