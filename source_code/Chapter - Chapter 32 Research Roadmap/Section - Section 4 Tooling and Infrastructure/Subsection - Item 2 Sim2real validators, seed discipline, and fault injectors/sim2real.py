#!/usr/bin/env python3
import json, hashlib, random, numpy as np
# deterministic seed mapping to scenario params
def seed_to_params(seed:int):
    h = hashlib.sha256(str(seed).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8],'big'))
    return {'target_speed': float(rng.uniform(1.0,5.0)),
            'target_heading': float(rng.uniform(-np.pi,np.pi))}
# simple linear dynamics + measurement; inject bias/dropout per plan
def simulate(params, seed, fault_plan):
    np.random.seed(seed); random.seed(seed)
    dt=0.1; T=50
    x=np.array([0.0,0.0,params['target_speed']]) # posx,posy,speed
    traj=[]; meas=[]
    for t in range(T):
        # motion (deterministic given seed and params)
        x[:2] += dt * params['target_speed'] * np.array([np.cos(params['target_heading']), np.sin(params['target_heading'])])
        traj.append(x.copy())
        z = x[:2] + np.random.normal(0,0.5,2) # noisy position
        # fault injection: at scheduled time inject bias or dropout
        if fault_plan.get('drop_at')==t:
            z = None
        if fault_plan.get('bias_at')==t:
            z = z + np.array([fault_plan['bias_x'], fault_plan['bias_y']])
        meas.append(None if z is None else z.tolist())
    return np.array(traj), meas
# simple KF estimate and NEES
def kalman_nees(traj, meas):
    n=2; P=np.eye(n)*1.0; xhat=np.zeros(n)
    Q=np.eye(n)*0.01; R=np.eye(n)*0.25
    nees_list=[]
    for k,(xt,z) in enumerate(zip(traj,meas)):
        # predict (identity motion omitted for brevity)
        P = P + Q
        if z is not None:
            y = z - xhat
            S = P + R
            K = P @ np.linalg.inv(S)
            xhat = xhat + K @ y
            P = (np.eye(n)-K) @ P
        err = xt[:2]-xhat
        nees_list.append(float(err.T @ np.linalg.inv(P) @ err))
    return np.mean(nees_list)
if __name__=='__main__':
    seed=42
    params=seed_to_params(seed)
    fault_plan={'drop_at':20,'bias_at':35,'bias_x':2.0,'bias_y':-1.0}
    traj,meas=simulate(params,seed,fault_plan)
    nees=kalman_nees(traj,meas)
    # persist artifacts for registry and CI trace
    out={'seed':seed,'params':params,'fault_plan':fault_plan,'nees':nees}
    print(json.dumps(out,indent=2))