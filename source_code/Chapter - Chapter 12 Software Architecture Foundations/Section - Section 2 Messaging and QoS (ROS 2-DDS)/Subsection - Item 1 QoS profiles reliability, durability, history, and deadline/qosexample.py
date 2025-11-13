#!/usr/bin/env python3
# Run with ROS 2 (rclpy). Demonstrates QoSProfile and deadline monitoring.
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor

class TrackPublisher(Node):
    def __init__(self):
        super().__init__('track_publisher')
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        qos.deadline = Duration(sec=0, nanosec=200_000_000)  # 0.2s contract
        self.pub = self.create_publisher(String, 'tracks', qos)
        self.count = 0
        self.create_timer(0.1, self.timer_cb)  # publish every 0.1s

    def timer_cb(self):
        m = String()
        m.data = f"track_update {self.count}"
        self.pub.publish(m)
        self.count += 1

class TrackSubscriber(Node):
    def __init__(self):
        super().__init__('track_subscriber')
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        qos.deadline = Duration(sec=0, nanosec=200_000_000)  # client-side expectation
        self.sub = self.create_subscription(String, 'tracks', self.cb, qos)
        self.last_recv = None
        self.deadline_secs = 0.2

    def cb(self, msg):
        now = time.time()
        if self.last_recv is not None:
            L = now - self.last_recv
            if L > self.deadline_secs:
                self.get_logger().warn(f"Deadline missed: inter-arrival {L:.3f}s > {self.deadline_secs}s")
        self.last_recv = now
        # process message (fast path): fusion update or enqueue for cognition

def main(args=None):
    rclpy.init(args=args)
    pub = TrackPublisher()
    sub = TrackSubscriber()
    exec = MultiThreadedExecutor()
    exec.add_node(pub); exec.add_node(sub)
    try:
        exec.spin()
    finally:
        exec.shutdown()
        pub.destroy_node(); sub.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()