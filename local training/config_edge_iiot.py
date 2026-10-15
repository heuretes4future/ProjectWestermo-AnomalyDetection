EDGE_IIOT_CONFIG = {
    "files": {
        "Node1": "data_edge_IIoT/output_node1.csv",
        "Node2": "data_edge_IIoT/output_node2.csv",
        "Node3": "data_edge_IIoT/output_node3.csv",
    },

    "columns_to_remove": [
        "frame.time",
        "ip.src_host", "ip.dst_host",
        "arp.dst.proto_ipv4", "arp.src.proto_ipv4",
        "http.file_data", "http.request.full_uri", "http.request.uri.query",
        "icmp.transmit_timestamp",
        "tcp.options", "tcp.payload",
        "tcp.srcport", "tcp.dstport", "udp.port",
        "mqtt.msg",

        "http.request.method",
        "http.referer",
        "http.request.version",
        "mqtt.protoname",
        "mqtt.topic",

        "icmp.unused",
        "http.tls_port",
        "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
        "mqtt.msg_decoded_as",
        "dns.retransmit_request_in",
        "dns.retransmit_request",
        "dns.retransmission",
        "mqtt.conack.flags",
    ],

    "categorical_columns": {},

    "force_numeric": [
        "dns.qry.name.len",
    ],

    "binary_label": "Attack_label",
    "multiclass_label": "Attack_type",
    "labels_to_drop": [],
    "merge_to_normal": [],
}
