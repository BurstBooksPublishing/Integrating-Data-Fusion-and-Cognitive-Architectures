#!/usr/bin/env python3
import time
from rclpy.node import Node
import rclpy
from rcl_interfaces.msg import SetParametersResult

class ConfigManager(Node):
    def __init__(self):
        super().__init__('config_manager')
        # declare control parameters
        self.declare_parameter('policy', 'baseline')        # current policy
        self.declare_parameter('safety.lock', False)       # safety lock
        self.declare_parameter('policy_dwell_ms', 5000)    # minimal dwell
        self._last_policy_time = 0.0
        # register validator callback for atomic validation
        self.add_on_set_parameters_callback(self._on_set_params)
        self.get_logger().info('Config manager ready')

    def _on_set_params(self, params):
        # build a dict of proposed changes
        props = {p.name: p.value for p in params}
        # check safety lock
        if props.get('safety.lock', self.get_parameter('safety.lock').value):
            # if locking, reject changes to policy or dwell
            if any(k.startswith('policy') for k in props.keys()):
                self.get_logger().warn('Rejecting policy change: safety.lock active')
                return SetParametersResult(successful=False, reason='safety.lock engaged')
        # enforce dwell for policy switches
        if 'policy' in props:
            now = time.time()
            last = self._last_policy_time
            T_d = self.get_parameter('policy_dwell_ms').value / 1000.0
            if (now - last) < T_d:
                self.get_logger().warn('Rejecting policy change: dwell time violation')
                return SetParametersResult(successful=False, reason='dwell_violation')
            # accept and update last time
            self._last_policy_time = now
            self.get_logger().info(f'Policy change accepted -> {props["policy"]}')
        return SetParametersResult(successful=True, reason='ok')

def main(args=None):
    rclpy.init(args=args)
    node = ConfigManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Usage: ros2 param set /config_manager policy aggressive
# Safety: if safety.lock true, that will be rejected.