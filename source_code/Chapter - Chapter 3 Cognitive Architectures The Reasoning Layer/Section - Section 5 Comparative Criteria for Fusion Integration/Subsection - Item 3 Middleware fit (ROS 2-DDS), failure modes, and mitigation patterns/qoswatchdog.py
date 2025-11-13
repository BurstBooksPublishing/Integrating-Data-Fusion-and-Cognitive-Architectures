#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, Deadline
from std_msgs.msg import String
from example_interfaces.srv import Trigger  # service to ask cognition to abstain

class TrackSubscriber(Node):
    def __init__(self):
        super().__init__('track_subscriber')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            durability=DurabilityPolicy.VOLATILE
        )
        qos.deadline = rclpy.duration.Duration(seconds=0, nanoseconds=200_000_000)  # 200ms deadline
        self.sub = self.create_subscription(String, 'tracks', self.cb_track, qos)
        self.last_seq = -1
        self.deadline_timer = self.create_timer(0.1, self._check_deadline)  # 100ms poll
        self.cog_client = self.create_client(Trigger, 'cognition/enter_degraded')
        self.get_logger().info('Track subscriber ready')

    def cb_track(self, msg):
        # update sequence/timebookkeeping for deadline monitoring
        self.last_seq = int(msg.data.split(':',1)[0])  # simple seq in payload
        self.get_logger().info(f'Received seq {self.last_seq}')

    def _check_deadline(self):
        # detect missed updates -> request cognitive fallback via service
        if self.last_seq < 0:
            return
        # simple staleness check: if no new seq in 400ms, declare deadline miss
        # (replace with DDS deadline/liveliness event handlers for production)
        now = self.get_clock().now().nanoseconds
        # store/update last_seen_time in production code
        if now - self.get_clock().now().nanoseconds > 400_000_000:
            if self.cog_client.wait_for_service(timeout_sec=0.5):
                req = Trigger.Request()  # trigger degraded mode
                self.cog_client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = TrackSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()