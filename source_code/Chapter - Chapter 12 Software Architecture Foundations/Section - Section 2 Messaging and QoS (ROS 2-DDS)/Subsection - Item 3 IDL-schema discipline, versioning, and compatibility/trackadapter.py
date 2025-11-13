#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
# Adapter node converts legacy track schema to extended schema with provenance.

class TrackAdapter(Node):
    def __init__(self):
        super().__init__('track_adapter')
        self.sub = self.create_subscription(
            String, '/track_v1', self.cb_v1, 10)  # QoS set by launch overrides
        self.pub = self.create_publisher(String, '/track_v2', 10)

    def cb_v1(self, msg):
        legacy = json.loads(msg.data)  # {id, x, y, vx}
        # Map fields: add uncertainty, timestamp, provenance, and optional fields.
        converted = {
            "track_id": legacy.get("id"),
            "pose": {"x": legacy.get("x"), "y": legacy.get("y")},
            "velocity": legacy.get("vx", 0.0),
            "covariance": legacy.get("cov", [[1.0,0],[0,1.0]]),  # default cov
            "timestamp": self.get_clock().now().to_msg().sec,  # monotonic stamp
            "provenance": {"source": "legacy_track_v1", "schema_version":"1.0.0"}
        }
        self.pub.publish(String(data=json.dumps(converted)))

def main(args=None):
    rclpy.init(args=args)
    node = TrackAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()