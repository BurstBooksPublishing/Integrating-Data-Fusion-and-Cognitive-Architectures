#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace, Node

# Base parameters and QoS intent (safe defaults)
BASE_PARAMS = {
    'fusion': {'update_rate_hz': 10},
    'qos_intent': {'reliability': 'reliable', 'history': 'keep_last', 'depth': 5,
                   'deadline_ms': 200}
}

# Small environment overlays keyed by env tag
ENV_OVERRIDES = {
    'dev': {'fusion': {'update_rate_hz': 5},
            'qos_intent': {'reliability': 'best_effort', 'depth': 1}},
    'prod': {'fusion': {'update_rate_hz': 20},
             'qos_intent': {'reliability': 'reliable', 'depth': 10, 'deadline_ms': 100}}
}

env = os.environ.get('COG_FUSION_ENV', 'dev')  # use \lstinline|COG_FUSION_ENV| in docs
params = BASE_PARAMS.copy()
params.update(ENV_OVERRIDES.get(env, {}))  # deterministic overlay

# Topology: group nodes under a fleet namespace and launch lifecycle-aware nodes
ns = 'robot_1'
ld = LaunchDescription([
    PushRosNamespace(namespace=ns),
    Node(package='cog_fusion', executable='fusion_node', name='fusion',
         parameters=[params], output='screen'),
    Node(package='cog_cognition', executable='reasoner', name='reasoner',
         parameters=[{'max_concurrent_hypotheses': 8}], output='screen'),
])
return ld