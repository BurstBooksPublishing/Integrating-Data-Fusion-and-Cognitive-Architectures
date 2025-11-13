#!/usr/bin/env python3
# Simple ROS2 lifecycle orchestrator (rclpy). Requires lifecycle nodes exposing GetState.
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import GetState, ChangeState
from lifecycle_msgs.msg import Transition
from collections import deque

# dependency graph: node_name -> list of hard deps
GRAPH = {
    'sensor_preproc': [],
    'track_manager': ['sensor_preproc'],
    'situation_assessor': ['track_manager'],
    'cognitive_reasoner': ['situation_assessor'],  # cognition waits for L2
}

TIMEOUT = 5.0  # seconds per probe

class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator')
        self.client_get = {}   # cache GetState clients
        self.client_change = {}
        for n in GRAPH:
            self.client_get[n] = self.create_client(GetState, f'/{n}/get_state')
            self.client_change[n] = self.create_client(ChangeState, f'/{n}/change_state')
        rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_service(self, client, node, timeout=TIMEOUT):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f'service for {node} unavailable')
            return False
        return True

    def get_state(self, node):
        cli = self.client_get[node]
        if not self.wait_for_service(cli, node): return None
        req = GetState.Request()
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=TIMEOUT)
        if fut.done() and fut.result():
            return fut.result().current_state.id
        return None

    def activate(self, node):
        cli = self.client_change[node]
        if not self.wait_for_service(cli, node): return False
        req = ChangeState.Request()
        req.transition.id = Transition.TRANSITION_ACTIVATE
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=TIMEOUT)
        return fut.done() and fut.result() is not None

    def topological_order(self):
        indeg = {n:0 for n in GRAPH}
        for n, deps in GRAPH.items():
            for d in deps: indeg[n]+=1
        q = deque([n for n,d in indeg.items() if d==0])
        order=[]
        while q:
            u=q.popleft(); order.append(u)
            for v, deps in GRAPH.items():
                if u in deps:
                    indeg[v]-=1
                    if indeg[v]==0: q.append(v)
        if len(order)!=len(GRAPH):
            raise RuntimeError('Cycle detected in dependency graph')
        return order

    def orchestrate(self):
        order = self.topological_order()
        for node in order:
            # wait for hard deps active
            ok=True
            for d in GRAPH[node]:
                state = self.get_state(d)
                ok = ok and (state == 3)  # 3 == ACTIVE per lifecycle_msgs
            if not ok:
                self.get_logger().error(f'dependencies for {node} not active')
                return False
            # local readiness probe could be added here (e.g., //health)
            if not self.activate(node):
                self.get_logger().error(f'failed to activate {node}')
                return False
            self.get_logger().info(f'{node} activated')
        return True

def main():
    rclpy.init()
    orch = Orchestrator()
    try:
        success = orch.orchestrate()
    finally:
        orch.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()