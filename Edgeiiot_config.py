from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EDGEIIOT_CONFIG = {
    "files": {
        "Node1": str(DATA_DIR / "output_node1.csv"),
        "Node2": str(DATA_DIR / "output_node2.csv"),
        "Node3": str(DATA_DIR / "output_node3.csv"),
    },

    "binary_label": "Attack_label",
    "multiclass_label": "Attack_type",

    "merge_to_normal": [],
    "labels_to_drop": [],

    "columns_to_remove": [
        "ip.src_host",
        "ip.dst_host",
        "frame.time",
        "frame.time_epoch",
        "http.file_data",
        "http.request.full_uri",
        "http.request.uri.query",
        "http.request.method",
        "http.referer",
        "http.request.version",
        "tcp.payload",
        "tcp.options",
        "mqtt.msg",
        "mqtt.protoname",
        "mqtt.topic",
        "arp.dst.proto_ipv4",
        "arp.src.proto_ipv4",
    ],

    "categorical_columns": {},

    "force_numeric": [
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "dns.qry.name.len",
        "mqtt.conack.flags",
    ],
}