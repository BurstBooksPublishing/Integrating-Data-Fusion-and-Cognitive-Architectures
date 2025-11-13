import numpy as np
import cv2

# policy parameters (could be loaded from secure store)
r_min = 0.6           # nominal comfort radius (m)
k = 2.0               # uncertainty multiplier
tau = 0.3             # ID probability threshold for redaction

def safe_distance(sigma_p):                # Eq. (1)
    return r_min + k * sigma_p

def id_probability(frame):                 # lightweight on-device check
    # placeholder: run small face-detector -> return probability in [0,1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rects = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return min(1.0, 0.2 * len(rects))     # coarse mapping to probability

def privacy_proximity_gate(track, frame):
    # track: dict with 'pos' (x,y), 'sigma_p' (m), 'consent' (bool)
    d = np.linalg.norm(track['pos'])       # distance from robot base
    r_safe = safe_distance(track['sigma_p'])
    if d < r_safe and not track.get('consent', False):
        # too close without consent: stop and request consent
        return {'action':'request_consent', 'log':'approach_blocked'}
    # privacy redaction: if frame likely contains ID and policy forbids capture
    p_id = id_probability(frame)
    if p_id >= tau and not track.get('consent_camera', False):
        # redact or replace with depth-only passthrough
        frame[:] = 0                         # simple redact (black-out) for demo
        return {'action':'continue_redacted', 'log':'frame_redacted'}
    return {'action':'continue', 'log':'ok'}

# Usage: called by cognition loop per sensor tick.