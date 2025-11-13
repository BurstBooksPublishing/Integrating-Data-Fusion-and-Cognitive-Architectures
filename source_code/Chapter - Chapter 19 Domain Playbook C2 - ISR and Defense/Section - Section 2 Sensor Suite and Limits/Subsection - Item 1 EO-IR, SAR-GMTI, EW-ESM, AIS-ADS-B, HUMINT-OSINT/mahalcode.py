import numpy as np

def mahalanobis_sq(z, x, H, S):
    # z: measurement (2,), x: state (n,), H: (2,n), S: innovation cov (2,2)
    v = z - H.dot(x)
    return float(v.T.dot(np.linalg.inv(S)).dot(v))

def cross_modal_score(mahal_sq, ais_auth_score, esm_bearing_mismatch):
    # combine numeric gate, AIS semantic score, and ESM cue into 0..1 trust
    num_score = np.exp(-0.5 * mahal_sq)          # from Mahalanobis
    esm_pen = np.exp(- (esm_bearing_mismatch/5.0)) # larger mismatch -> smaller
    combined = 0.6 * num_score + 0.3 * ais_auth_score + 0.1 * esm_pen
    return combined

# Example inputs
z = np.array([1000., 200.])    # radar-measured (x,y) meters
x = np.array([998., 202., 0.]) # state estimate (x,y,vel)
H = np.array([[1,0,0],[0,1,0]])
S = np.array([[25., 0.],[0., 25.]]) # innovation cov
mahal_sq = mahalanobis_sq(z, x, H, S)
ais_auth = 0.4                  # low AIS authenticity (0..1)
esm_mismatch_deg = 20.0
score = cross_modal_score(mahal_sq, ais_auth, esm_mismatch_deg)
# Decision thresholding
if score > 0.5:
    print("Associate AIS↔radar (score=%.2f)"%score)
else:
    print("Flag mismatch; promote spoof hypothesis (score=%.2f)"%score)