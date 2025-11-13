import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, Duration

def qos_for_tier(tier):
    # tier: 'A' (control), 'B' (tracks), 'C' (telemetry)
    if tier == 'A':
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            deadline=Duration(seconds=0, nanoseconds=5_000_000) # 5 ms
        )
    if tier == 'B':
        # depth computed offline by eq. (1); here use a parameterized default
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            deadline=Duration(seconds=0, nanoseconds=50_000_000) # 50 ms
        )
    # tier C: telemetry
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_ALL,
        depth=0
    )

def main():
    rclpy.init()
    node = Node('qos_demo')
    pubA = node.create_publisher(String, 'act_cmd', qos_for_tier('A')) # control
    pubB = node.create_publisher(String, 'tracks', qos_for_tier('B'))  # tracks
    pubC = node.create_publisher(String, 'telemetry', qos_for_tier('C')) # logs
    # publish loop omitted for brevity
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()