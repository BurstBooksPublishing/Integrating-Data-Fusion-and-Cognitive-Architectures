import numpy as np
from scipy.signal import wiener
# simulate range profile with a target peak
np.random.seed(1)
N=512
noise_std=1.0
profile = np.random.normal(0,noise_std,size=N)
profile[250] += 12.0  # target
# denoise (Wiener)
profile_filt = wiener(profile, mysize=11)
# simple cell-averaging CFAR
def cfar_ca(x, guard=2, ref=16, alpha=4.0):
    N=len(x); detected=np.zeros(N,bool)
    for i in range(ref+guard, N-ref-guard):
        left = x[i-ref-guard:i-guard]
        right = x[i+guard+1:i+guard+ref+1]
        noise_est = (np.sum(left)+np.sum(right))/(2*ref)
        T = alpha*noise_est
        detected[i] = x[i] > T
    return detected
detected = cfar_ca(profile_filt)
# extrinsic calibration: rotation + translation
def apply_extrinsic(pts, R, t):
    return (R @ pts.T).T + t
# create two noisy 3D point clouds with known transform
M=50
A = np.random.randn(M,3) + np.array([0,0,5])  # sensor A points
R_true = np.array([[0.99, -0.05, 0.0],[0.05,0.99,0.0],[0.0,0.0,1.0]])
t_true = np.array([0.5,-0.2,0.1])
B = apply_extrinsic(A, R_true, t_true) + 0.01*np.random.randn(M,3)
# compute rigid transform via SVD (Umeyama/Procrustes)
def rigid_transform(A_pts, B_pts):
    mu_A = A_pts.mean(axis=0); mu_B=B_pts.mean(axis=0)
    A0 = A_pts - mu_A; B0 = B_pts - mu_B
    U,S,Vt = np.linalg.svd(A0.T @ B0)
    R = Vt.T @ U.T
    if np.linalg.det(R)<0: Vt[-1,:]*=-1; R=Vt.T@U.T
    t = mu_B - R @ mu_A
    return R,t
R_est, t_est = rigid_transform(A,B)
# residuals and diagnostics
res = B - apply_extrinsic(A, R_est, t_est)
rms = np.sqrt((res**2).mean())
print("Detected peak index:", np.where(detected)[0])
print("Registration RMS error:", rms)  # diagnostic