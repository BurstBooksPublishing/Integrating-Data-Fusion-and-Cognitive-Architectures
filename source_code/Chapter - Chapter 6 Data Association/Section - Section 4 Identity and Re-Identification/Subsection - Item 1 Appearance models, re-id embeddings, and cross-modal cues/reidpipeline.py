import numpy as np

def cos_sim(a,b): return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)  # cosine

def update_track_embedding(track_emb, obs_emb, alpha=0.2):
    # EMA update for track descriptor (keeps memory and resists brief occlusion)
    return (1-alpha)*track_emb + alpha*obs_emb

def motion_loglikelihood(track_state, obs_pos, S_inv):
    # Gaussian log-likelihood under predicted track state covariance inverse
    residual = obs_pos - track_state['pred_pos']
    return -0.5 * residual.T @ S_inv @ residual  # unnormalized log-lik

def association_scores(tracks, observations, S_inv, alpha=0.6):
    scores = np.full((len(tracks), len(observations)), -np.inf)
    for i,t in enumerate(tracks):
        for j,o in enumerate(observations):
            a_sim = cos_sim(o['emb'], t['emb'])          # appearance similarity
            m_log = motion_loglikelihood(t, o['pos'], S_inv)  # motion score
            scores[i,j] = alpha*a_sim + (1-alpha)*m_log
    return scores

# Minimal runnable example: one track, one observation
if __name__=='__main__':
    track = {'emb': np.random.randn(128), 'pred_pos': np.array([0.,0.])}
    obs = {'emb': np.random.randn(128), 'pos': np.array([0.1, -0.05])}
    S_inv = np.eye(2)*100.0  # tight gating
    s = association_scores([track],[obs],S_inv)[0,0]
    track['emb'] = update_track_embedding(track['emb'], obs['emb'])  # update
    print("score", s)