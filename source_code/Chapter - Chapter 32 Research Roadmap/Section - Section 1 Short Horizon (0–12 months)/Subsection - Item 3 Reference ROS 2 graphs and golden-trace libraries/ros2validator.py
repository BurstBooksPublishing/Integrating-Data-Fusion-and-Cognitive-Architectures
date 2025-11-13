#!/usr/bin/env python3
# Simple ROS 2 graph validator; starts ros2 bag on conformance.
import rclpy, subprocess, yaml, sys
from rclpy.node import Node

class GraphValidator(Node):
    def __init__(self, spec_path):
        super().__init__('graph_validator') 
        with open(spec_path,'r') as f:
            self.spec = yaml.safe_load(f)     # reference graph spec
        self.timer = self.create_timer(1.0, self.check_graph)

    def check_graph(self):
        # query live graph
        nodes = self.get_node_names_and_namespaces()
        topics = self.get_topic_names_and_types()
        # simple conformity checks
        expected_nodes = set((n['name'],n.get('ns','/')) for n in self.spec['nodes'])
        live_nodes = set(nodes)
        if expected_nodes.issubset(live_nodes):
            self.get_logger().info('Graph matches spec; triggering record.')
            self.trigger_record()
            rclpy.shutdown()
        else:
            self.get_logger().info('Graph mismatch; retrying.')

    def trigger_record(self):
        # build ros2 bag record command from spec topics
        topic_list = [t['name'] for t in self.spec['topics']]
        cmd = ['ros2','bag','record','-o','golden_trace'] + topic_list
        # start recorder as subprocess; CI retains process logs
        subprocess.Popen(cmd)  # recorder runs until manually stopped

def main(argv=None):
    rclpy.init(args=argv)
    if len(sys.argv) < 2:
        print('Usage: validator.py reference_spec.yaml'); return
    node = GraphValidator(sys.argv[1])
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()