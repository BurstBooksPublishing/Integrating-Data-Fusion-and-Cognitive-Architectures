import numpy as np

# simple 3-state track: [x, y, yaw]; covariance P
def predict(x, P, u, Q, dt):
    # motion model: forward velocity u[0], yaw rate u[1]
    v, r = u
    yaw = x[2]
    x = x + np.array([v*np.cos(yaw)*dt, v*np.sin(yaw)*dt, r*dt])
    # Jacobian F
    F = np.eye(3)
    F[0,2] = -v*np.sin(yaw)*dt
    F[1,2] =  v*np.cos(yaw)*dt
    P = F @ P @ F.T + Q
    return x, P

def lateral_measurement(x, lane_poly):
    # compute lateral offset and heading residual to nearest lane segment
    # lane_poly: Nx2 array of polyline points
    # returns z, H (measurement jacobian)
    pos = x[:2]
    # naive nearest-segment: project onto polyline (omitted for brevity)
    # here use placeholder nearest point p and heading theta_l
    p = lane_poly[0]  # placeholder
    theta_l = 0.0     # placeholder
    dx = pos - p
    lateral = -np.sin(theta_l)*dx[0] + np.cos(theta_l)*dx[1]
    # approximate jacobian H = d(lateral)/d(x,y)
    H = np.array([[-np.sin(theta_l), np.cos(theta_l), 0.0]])
    return np.array([lateral]), H

def update(x, P, z, R, lane_poly):
    z_pred, H = lateral_measurement(x, lane_poly)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + (K @ (z - z_pred)).ravel()
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P

# example usage
x = np.array([0., 0., 0.]); P = np.eye(3)*0.5
Q = np.diag([0.1,0.1,0.01]); R = np.array([[0.2]])
u = (1.0, 0.01); dt = 0.1
lane_poly = np.array([[0.,0.],[10.,0.]])  # simple straight lane

x, P = predict(x, P, u, Q, dt)            # predict step
z = np.array([0.1])                       # measured lateral offset
x, P = update(x, P, z, R, lane_poly)      # lane fusion update
print(x, P)