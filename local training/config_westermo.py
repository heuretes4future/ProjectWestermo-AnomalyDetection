WESTERMO_CONFIG = {
    # Files per node
    "files": {
        "Right":  "data/output_right.csv",
        "Bottom": "data/output_bottom.csv",
        "Left":   "data/output_left.csv",
    },

    # Columns to remove (addresses and time-related)
    "columns_to_remove": [
        "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs",
        "startDate", "endDate", "start", "end", "startOffset", "endOffset",
    ],

    # Categorical columns and their known values
    "categorical_columns": {
        "protocol": ["ARP", "IPV4-TCP", "IPV4-UDP", "IPV4-2", "LLDP", "IPV4-ICMP"],
    },

    # Label configuration
    "binary_label": "NST_B_Label",
    "multiclass_label": "NST_M_Label",
    "labels_to_drop": ["IT_B_Label", "IT_M_Label"],

    # Classes to merge into Normal
    "merge_to_normal": ["Good-SSH", "GOOD-SSH"],
}
