import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

class TrackPublisher(Node):
    def __init__(self):
        super().__init__('track_pub')
        # Reliable, transient-local, keep-last depth=10, deadline 100ms, lifespan 500ms
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            deadline=Duration(seconds=0, nanoseconds=100_000_000),   # 100ms
            lifespan=Duration(seconds=0, nanoseconds=500_000_000)   # 500ms
        )
        self.pub = self.create_publisher(String, 'tracks', qos)
        self.timer = self.create_timer(0.05, self.publish_track)  # 20 Hz

    def publish_track(self):
        m = String()
        # embed provenance as JSON or typed fields in real schemas
        m.data = f'{{"ts":{self.get_clock().now().nanoseconds},"id":"trk1"}}'
        self.pub.publish(m)

class TrackSubscriber(Node):
    def __init__(self):
        super().__init__('track_sub')
        # must match or be compatible with publisher QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST,
                         deadline=Duration(seconds=0, nanoseconds=100_000_000))
        self.sub = self.create_subscription(String, 'tracks', self.cb, qos)
        # register deadline callback if available via lifecycle or monitors (platform-specific)
    def cb(self, msg):
        # validate timestamp/provenance and staleness
        self.get_logger().info(f'Recv: {msg.data[:120]}')

def main(args=None):
    rclpy.init(args=args)
    pub = TrackPublisher()
    sub = TrackSubscriber()
    try:
        rclpy.spin_once(pub, timeout_sec=0)  # integrate into executor in real system
    finally:
        pub.destroy_node()
        sub.destroy_node()
        rclpy.shutdown()