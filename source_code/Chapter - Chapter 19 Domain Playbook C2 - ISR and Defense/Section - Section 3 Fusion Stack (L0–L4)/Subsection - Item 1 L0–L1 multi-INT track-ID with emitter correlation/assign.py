import numpy as np
from scipy.optimize import linear_sum_assignment

def maha_cost(z, Hx, S):
    # Mahalanobis distance squared as cost component
    diff = z - Hx
    return float(diff.T @ np.linalg.inv(S) @ diff)

def emitter_score(cost_em): 
    # map emitter signature cost to a comparable assignment cost
    return -np.log(1e-6 + np.exp(-cost_em))  # higher match => lower cost

def build_cost_matrix(measurements, tracks, S_list, emitter_costs, alpha=1.0, beta=1.0):
    m = len(measurements); n = len(tracks)
    C = np.full((m,n), 1e6)  # large cost for infeasible
    for i,z in enumerate(measurements):
        for j,trk in enumerate(tracks):
            d2 = maha_cost(z['z'], trk['Hx'], S_list[j])
            if d2 < z.get('gate_thresh',9.21):  # 95% chi-square 3-dof ~7.82; example
                C[i,j] = alpha * d2 + beta * emitter_score(emitter_costs[i].get(trk['id'], 10.0))
    return C

# Example usage
measurements = [{'z':np.array([1000.,200.])}, {'z':np.array([1020.,210.])}]
tracks = [{'Hx':np.array([995.,205.]), 'id':'T1'}, {'Hx':np.array([300.,50.]), 'id':'T2'}]
S_list = [np.eye(2)*25., np.eye(2)*25.]
emitter_costs = [ {'T1':0.1,'T2':5.0}, {'T1':0.2,'T2':6.0} ]

C = build_cost_matrix(measurements, tracks, S_list, emitter_costs)
row_ind,col_ind = linear_sum_assignment(C)
assignments = list(zip(row_ind, col_ind))
print('assignments', assignments)  # pairs of (measurement, track index)