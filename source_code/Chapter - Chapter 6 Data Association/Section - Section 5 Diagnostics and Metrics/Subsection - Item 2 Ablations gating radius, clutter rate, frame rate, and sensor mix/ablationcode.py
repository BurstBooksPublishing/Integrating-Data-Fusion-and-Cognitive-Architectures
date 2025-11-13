import numpy as np
from scipy.stats import chi2, multivariate_normal
from scipy.optimize import linear_sum_assignment

# simple linear KF predict step (position only)
def predict_tracks(tracks, dt, q=0.01):
    F = np.array([[1, dt],[0,1]])  # state: pos, vel
    Q = q * np.eye(2)
    for t in tracks:
        t['x'] = F @ t['x']
        t['P'] = F @ t['P'] @ F.T + Q
    return tracks

# generate detections for true tracks + Poisson clutter
def gen_detections(tracks, p_det=0.9, lam_clutter=0.1, area=[-50,50,-50,50]):
    dets = []
    for t in tracks:
        if np.random.rand() < p_det:
            z = t['x'][0] + np.random.randn()*np.sqrt(t['P'][0,0])  # position meas
            dets.append(np.array([z]))
    # clutter
    area_size = (area[1]-area[0])*(area[3]-area[2])
    n_clutter = np.random.poisson(lam_clutter*area_size)
    for _ in range(n_clutter):
        dets.append(np.array([np.random.uniform(area[0],area[1])]))
    return dets

# Mahalanobis gating and cost matrix (1D position)
def gate_and_cost(tracks, dets, p_gate=0.95, R_factor=1.0):
    d=1
    R2 = chi2.ppf(p_gate, d) * R_factor
    m=len(tracks); n=len(dets)
    cost = 1e6*np.ones((m,n))
    gated_count=0
    for i,t in enumerate(tracks):
        S = np.array([[t['P'][0,0] + t['R']]])
        for j,z in enumerate(dets):
            inn = (z - t['x'][0])
            m2 = (inn.T @ np.linalg.inv(S) @ inn)
            if m2 <= R2:
                cost[i,j]=np.linalg.norm(inn)  # simple cost
                gated_count+=1
    return cost, gated_count

# minimal ID switch tracker update
def assign_and_update(tracks, dets, cost):
    row,col = linear_sum_assignment(cost)
    assigned = {}
    id_switches=0
    for r,c in zip(row,col):
        if cost[r,c] < 1e5:
            prev_id = tracks[r].get('assoc_id')
            tracks[r]['assoc_id'] = c
            if prev_id is not None and prev_id != c:
                id_switches += 1
    return id_switches

# example run
np.random.seed(1)
tracks=[{'x':np.array([i*5.0,0.0]), 'P':np.eye(2)*0.1, 'R':0.5, 'assoc_id':None} for i in range(3)]
tracks = predict_tracks(tracks, dt=0.1)
dets = gen_detections(tracks, lam_clutter=0.001)
cost,gated = gate_and_cost(tracks,dets,p_gate=0.95)
id_sw = assign_and_update(tracks,dets,cost)
print("dets",len(dets),"gated",gated,"id_switches",id_sw)