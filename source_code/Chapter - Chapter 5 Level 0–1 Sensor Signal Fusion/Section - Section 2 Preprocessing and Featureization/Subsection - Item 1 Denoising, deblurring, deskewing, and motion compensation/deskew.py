import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def deskew_pointcloud(points, point_times, pose_times, quats, translations, ref_time):
    """
    points: (N,3) array in sensor frame.
    point_times: (N,) timestamps for each point.
    pose_times: (M,) timestamps for poses (monotonic).
    quats: (M,4) rotation quaternions (xyzw) from sensor->world at pose_times.
    translations: (M,3) translations at pose_times (world frame).
    ref_time: scalar reference timestamp to map points to.
    returns: (N,3) deskewed points in world frame at ref_time.
    """
    # build rotation interpolator
    rots = Rotation.from_quat(quats)               # SciPy expects (x,y,z,w)
    slerp = Slerp(pose_times, rots)
    # sample rotation and translation at each point time and ref_time
    rot_pts = slerp(point_times)                    # Rotation objects per point
    rot_ref = slerp([ref_time])[0]
    # interpolate translations linearly
    translations = np.asarray(translations)
    trans_pts = np.vstack([np.interp(point_times, pose_times, translations[:,i]) 
                           for i in range(3)]).T
    trans_ref = np.interp(ref_time, pose_times, translations, axis=0)
    # transform each point to world at its time, then back to ref frame
    pts_world = rot_pts.apply(points) + trans_pts
    # compute inverse of transform at ref_time
    Rref = rot_ref.as_matrix()
    tref = trans_ref
    pts_ref = (Rref.T @ (pts_world - tref).T).T               # points in sensor frame at ref_time
    return pts_ref

# Example usage (synthetic): deskew half-second scan to start time.