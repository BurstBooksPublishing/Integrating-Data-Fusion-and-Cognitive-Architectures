import numpy as np

def gaussian_loglik(obs, mean, var):
    # obs: (T, D); mean: (D,), var: (D,) diag cov
    diff = obs - mean
    return -0.5 * (np.sum((diff**2) / var, axis=1)
                   + obs.shape[1]*np.log(2*np.pi) + np.sum(np.log(var)))

def segmental_viterbi(obs, states, start_logprob, trans_logprob, 
                      state_means, state_vars, duration_logprob, Dmax):
    T = obs.shape[0]; S = len(states)
    best = -np.inf * np.ones((T+1, S))  # best[t, s]: best score ending at t with state s
    prev = -np.ones((T+1, S), int)
    best[0,:] = start_logprob  # allow zero-length start scoring

    # precompute emission loglik for each state and time
    emis = np.stack([gaussian_loglik(obs, state_means[s], state_vars[s]) for s in range(S)], axis=1)
    # cumulative sums to get segment loglik quickly
    emis_cum = np.vstack([np.zeros((1,S)), np.cumsum(emis, axis=0)])

    for t in range(1, T+1):
        for s in range(S):
            # consider durations d ending at time t
            dmin = 1
            dmax = min(Dmax, t)
            dur_probs = duration_logprob[s, dmin:dmax+1]  # log-probs for durations
            for d_idx, d in enumerate(range(dmin, dmax+1)):
                t0 = t - d
                seg_emis = emis_cum[t, s] - emis_cum[t0, s]
                score_candidates = best[t0, :] + trans_logprob[:, s]  # from all previous states
                best_prev = np.max(score_candidates)
                total = best_prev + dur_probs[d_idx] + seg_emis
                if total > best[t, s]:
                    best[t, s] = total
                    prev[t, s] = np.argmax(score_candidates)

    # backtrack
    labels = []
    t = T
    cur_state = int(np.argmax(best[T, :]))
    while t > 0:
        # find duration by scanning backwards (costly but small Dmax typical)
        for d in range(1, min(Dmax, t)+1):
            t0 = t - d
            seg_emis = emis_cum[t, cur_state] - emis_cum[t0, cur_state]
            score_candidates = best[t0, :] + trans_logprob[:, cur_state]
            if np.max(score_candidates) + duration_logprob[cur_state, d] + seg_emis == best[t, cur_state]:
                labels.insert(0, (t0, t, cur_state))
                cur_state = int(np.argmax(score_candidates))
                t = t0
                break
        else:
            # fall back if match not found
            t -= 1
    return labels

# Usage with synthetic inputs omitted for brevity.