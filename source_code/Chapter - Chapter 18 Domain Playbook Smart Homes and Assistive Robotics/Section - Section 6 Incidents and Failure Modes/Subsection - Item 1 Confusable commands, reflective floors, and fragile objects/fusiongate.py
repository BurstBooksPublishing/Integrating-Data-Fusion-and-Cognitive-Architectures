import numpy as np

def entropy(p):
    p = np.asarray(p); p = p[p>0]
    return -np.sum(p*np.log(p))

def fuse_and_decide(asr_nbest, vision_objs, tactile, prior=None,
                    H_thresh=0.8):
    # asr_nbest: list of (intent, score)
    # vision_objs: list of detected object dicts
    # tactile: tactile reading or None
    intents, scores = zip(*asr_nbest)
    scores = np.array(scores); scores /= scores.sum()
    # simple likelihood: boost intents present in vision
    vis_lik = np.array([1.5 if any(o['label']==i for o in vision_objs) else 0.6
                        for i in intents])
    posterior = scores * vis_lik
    posterior /= posterior.sum()
    H = entropy(posterior)
    best = intents[np.argmax(posterior)]
    # fragile check
    fragile = any(o.get('fragile',False) and o['label']==best
                  for o in vision_objs)
    if H > H_thresh:
        return {'action':'clarify','candidates':intents,'entropy':H}
    if fragile and tactile is None:
        return {'action':'require_tactile_confirm','target':best}
    if fragile:
        return {'action':'execute_gentle_grasp','target':best}
    return {'action':'execute_normal_grasp','target':best}

# Example run (simulated)
asr = [('glass',0.55),('plate',0.45)]
vision = [{'label':'glass','fragile':True},{'label':'glass_mirror'}]
print(fuse_and_decide(asr,vision,None))