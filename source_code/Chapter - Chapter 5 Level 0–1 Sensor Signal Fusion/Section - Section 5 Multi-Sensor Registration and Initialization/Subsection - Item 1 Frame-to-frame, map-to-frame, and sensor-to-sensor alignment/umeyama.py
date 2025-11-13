import numpy as np

def umeyama(p, q):
    # p, q: (N,3) corresponding points (sensor, world)
    assert p.shape == q.shape
    N = p.shape[0]
    mp, mq = p.mean(axis=0), q.mean(axis=0)
    P = p - mp; Q = q - mq
    C = Q.T @ P / N
    U, S, Vt = np.linalg.svd(C)
    D = np.diag([1,1,np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    t = mq - R @ mp
    return R, t

# example usage
if __name__ == "__main__":
    # generate synthetic transform and noisy correspondences
    np.random.seed(0)
    R_true = np.eye(3)  # replace with a test rotation
    t_true = np.array([0.5, -0.2, 0.3])
    p = np.random.randn(100,3)
    q = (R_true @ p.T).T + t_true + 0.01*np.random.randn(100,3)  # observed in world
    R_est, t_est = umeyama(p, q)
    q_hat = (R_est @ p.T).T + t_est
    rmse = np.sqrt(np.mean(np.sum((q - q_hat)**2, axis=1)))
    print("RMSE:", rmse)  # diagnostic