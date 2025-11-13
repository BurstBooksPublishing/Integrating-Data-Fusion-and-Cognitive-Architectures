#!/usr/bin/env python3
# Minimal ROS 2 node that broadcasts a namespaced static TF and looks it up.
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

class NamespacedTfNode(Node):
    def __init__(self):
        super().__init__('ns_tf_node')
        # use node's namespace to build frame names (works when launched with PushRosNamespace)
        ns = self.get_namespace().lstrip('/') or 'robot0'
        self.map_frame = f'/{ns}/map'            # namespaced map
        self.base_frame = f'/{ns}/base_link'     # namespaced base
        # static broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame      # explicit namespaced frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)       # publish once (static)
        # dynamic listener/lookup example
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # small timer to attempt lookup
        self.create_timer(0.5, self.try_lookup)

    def try_lookup(self):
        try:
            # lookup using the exact namespaced identifiers
            trans = self.tf_buffer.lookup_transform(self.map_frame,
                                                   self.base_frame,
                                                   Time())      # latest available
            self.get_logger().info(f'Lookup ok: {trans.header.frame_id} -> {trans.child_frame_id}')
        except Exception as e:
            self.get_logger().warn(f'Waiting for TF: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = NamespacedTfNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Run with: ros2 run  tf_namespaced --ros-args -r __ns:=/robot1