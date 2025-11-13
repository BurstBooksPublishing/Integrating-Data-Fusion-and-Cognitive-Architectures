import numpy as np
from scipy.spatial.transform import Rotation as R

def slerp_rot(r0, r1, t):
    # r0,r1 are scipy Rotation; t scalar in [0,1]
    return R.slerp(0, 1, [r0, r1])(t)

def interp_pose(poses, times, t_query):
    # poses: list of (trans (3,), rot as Rotation), times: array
    idx = np.searchsorted(times, t_query)
    if idx == 0: return poses[0]
    if idx >= len(times): return poses[-1]
    t0, t1 = times[idx-1], times[idx]
    w = (t_query - t0) / (t1 - t0)
    p0, r0 = poses[idx-1]
    p1, r1 = poses[idx]
    trans = (1-w)*p0 + w*p1                          # linear interp translation
    rot = slerp_rot(r0, r1, w)                       # slerp rotation
    return trans, rot

def transform_points_to_world(points_sensor, times_points, poses, times_poses, T_ego_sensor):
    # points_sensor: (N,3), times_points: (N,), poses: list[(3,),Rotation], T_ego_sensor: (4,4)
    N = points_sensor.shape[0]
    pts_world = np.empty((N,3))
    for i in range(N):
        p_s = np.hstack([points_sensor[i], 1.0])
        t_query = times_points[i]
        trans_ego, rot_ego = interp_pose(poses, times_poses, t_query)
        T_world_ego = np.eye(4)
        T_world_ego[:3,:3] = rot_ego.as_matrix()     # rotation from ego to world
        T_world_ego[:3,3]  = trans_ego               # translation ego->world
        p_w = T_world_ego @ (T_ego_sensor @ p_s)     # compose transforms
        pts_world[i] = p_w[:3]
    return pts_world