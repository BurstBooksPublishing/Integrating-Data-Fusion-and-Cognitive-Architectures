import numpy as np
from numpy.linalg import inv, LinAlgError

def mahalanobis2(xi, xj, Si, Sj):
    S = Si + Sj
    try:
        invS = inv(S)
    except LinAlgError:
        invS = np.linalg.pinv(S)  # robust fallback
    d2 = float((xi - xj).T @ invS @ (xi - xj))
    return d2

def consensusDecision(reports, vetoThreshold=9.21, consensusThresh=0.6):
    # reports: list of dicts: {'id':str, 'x':np.array, 'S':np.array, 'provenance':str}
    n = len(reports)
    # compute weights inversely proportional to trace of covariance
    traces = np.array([np.trace(r['S']) + 1e-6 for r in reports])
    weights = 1.0 / traces
    weights /= weights.sum()
    # weighted consensus mean
    xstack = np.stack([r['x'] for r in reports], axis=0)
    xbar = (weights[:,None] * xstack).sum(axis=0)
    # pairwise Mahalanobis checks
    pairwise = {}
    maxD2 = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d2 = mahalanobis2(reports[i]['x'], reports[j]['x'],
                              reports[i]['S'], reports[j]['S'])
            pairwise[(reports[i]['id'], reports[j]['id'])] = d2
            maxD2 = max(maxD2, d2)
    # consensus confidence proxy: inverse weighted mean trace
    consensusConf = 1.0 / (np.dot(weights, traces))
    # decision logic: veto on strong inconsistency else accept/defer
    if maxD2 > vetoThreshold:
        action = 'veto'           # hard disagreement
    elif consensusConf > consensusThresh:
        action = 'accept'         # corroborated
    else:
        action = 'abstain'        # insufficient confidence
    # rationale record for audit
    rationale = {
        'consensus_mean': xbar,
        'consensus_confidence': float(consensusConf),
        'max_pairwise_d2': float(maxD2),
        'pairwise': pairwise,
        'weights': weights.tolist(),
        'action': action,
        'evidence_ids': [r['id'] for r in reports],
        'provenances': [r['provenance'] for r in reports]
    }
    return rationale

# Example usage (2D position reports)
if __name__ == "__main__":
    r1 = {'id':'camA','x':np.array([10.0,5.0]),'S':np.eye(2)*0.5,'provenance':'camA@ts1'}
    r2 = {'id':'lidar1','x':np.array([9.8,5.1]),'S':np.eye(2)*0.2,'provenance':'lidar1@ts1'}
    r3 = {'id':'radarX','x':np.array([15.0,8.0]),'S':np.eye(2)*2.0,'provenance':'radarX@ts1'}  # divergent
    print(consensusDecision([r1,r2,r3]))