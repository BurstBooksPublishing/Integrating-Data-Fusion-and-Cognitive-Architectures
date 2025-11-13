from datetime import datetime, timedelta

# Example track structure: {'id':str, 'identity_linked':bool, 'consent_expiry':datetime or None, 'modalities':{'video':True,...}, 'features':{...}}
E_MAX = 0.3  # threshold for exposure metric

def exposure_score(track):
    # simple heuristic: identity linkage and modality sensitivity raise exposure
    score = 0.0
    if track.get('identity_linked'): score += 0.6
    if track.get('modalities',{}).get('video'): score += 0.3
    if track.get('modalities',{}).get('audio'): score += 0.2
    return min(1.0, score)

def redact_track(track):
    # apply minimization and respect consent
    now = datetime.utcnow()
    consent_ok = track.get('consent_expiry') and track['consent_expiry'] > now
    e = exposure_score(track)
    redacted = track.copy()
    redacted['exposure_score'] = e
    if not consent_ok or e > E_MAX:
        # remove raw artifacts and identity fields
        redacted.pop('raw_frames', None)
        redacted['identity_linked'] = False
        # collapse features into privacy-preserving summary
        feats = redacted.get('features',{})
        redacted['features'] = {'summary': sum(feats.values()) if feats else None}
        action = 'redacted'
    else:
        action = 'kept'
    # audit entry
    audit = {'track_id': track['id'], 'time': now.isoformat(), 'action': action, 'exposure': e}
    return redacted, audit

# Example usage
if __name__ == "__main__":
    t = {'id':'T123','identity_linked':True,'consent_expiry':datetime.utcnow()+timedelta(hours=1),
         'modalities':{'video':True,'lidar':False},'features':{'speed':0.9}, 'raw_frames':'...'}
    r,a = redact_track(t)
    print(r)  # process-safe track
    print(a)  # append to secure audit log