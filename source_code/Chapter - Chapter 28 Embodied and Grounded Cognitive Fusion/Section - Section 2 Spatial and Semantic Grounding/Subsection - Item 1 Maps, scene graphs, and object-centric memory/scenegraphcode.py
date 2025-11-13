import time, numpy as np

def kalman_update(x, P, z, R, H=np.eye(3)):
    # Linear Kalman update for 3D pose vector (x,y,theta)
    Pm = P  # prior
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    x_up = x + K @ (z - H @ x)
    P_up = (np.eye(len(x)) - K @ H) @ Pm
    return x_up, P_up

class SceneGraph:
    def __init__(self):
        self.nodes = {}  # id -> dict(attrs)
        self.edges = set()  # (id1,id2,relation)

    def add_or_update(self, node_id, pose, cov, meas_source):
        now = time.time()
        if node_id in self.nodes:
            n = self.nodes[node_id]
            n['pose'], n['cov'] = kalman_update(n['pose'], n['cov'], pose, cov)
            n['last_seen'] = now
            n['provenance'].append((meas_source, now))
        else:
            self.nodes[node_id] = {
                'pose': pose.copy(), 'cov': cov.copy(),
                'created': now, 'last_seen': now,
                'provenance': [(meas_source, now)]
            }

    def query_by_region(self, center, radius):
        out=[]
        for nid,n in self.nodes.items():
            d = np.linalg.norm(n['pose'][:2]-center[:2])
            if d<=radius: out.append(nid)
        return out

# Example usage
sg = SceneGraph()
pose = np.array([1.0, 2.0, 0.0])
cov = np.diag([0.1,0.1,0.01])
sg.add_or_update('chair_1', pose, cov, 'vision_cam_1')
# later measurement with small displacement
sg.add_or_update('chair_1', np.array([1.1,2.05,0.01]), cov, 'vision_cam_2')
print('Chair pose', sg.nodes['chair_1']['pose'])  # fused pose