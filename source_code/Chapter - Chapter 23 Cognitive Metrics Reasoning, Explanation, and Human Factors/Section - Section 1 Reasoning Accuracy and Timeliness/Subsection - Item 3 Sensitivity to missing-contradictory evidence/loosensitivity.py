import numpy as np

def loo_sensitivity(evidence_list, score_fn):
    """
    evidence_list: list of evidence objects
    score_fn: callable(E) -> dict{hypothesis: log_prob}
    returns: dict{hypothesis: np.array of delta per evidence}
    """
    # baseline log-posteriors
    base = score_fn(evidence_list)                      # dict of log P(H|E)
    hypotheses = list(base.keys())
    n = len(evidence_list)
    deltas = {h: np.zeros(n, dtype=float) for h in hypotheses}
    # compute LOO scores
    for i in range(n):
        E_minus = evidence_list[:i] + evidence_list[i+1:]
        loo = score_fn(E_minus)                         # log P(H|E\{e_i})
        for h in hypotheses:
            deltas[h][i] = base[h] - loo.get(h, -np.inf) # eq. (1)
    return deltas

# Example score_fn: stub using softmax over heuristic log-likelihoods
def score_fn_stub(E):
    # E: list of dicts with per-hypothesis log-likelihoods
    loglikes = {}
    for h in ["H1","H2","H3"]:
        loglikes[h] = sum(e.get(h, -10.0) for e in E)  # sum log-likelihoods
    # normalize to log-posteriors (uniform prior)
    logvals = np.array([loglikes[h] for h in ["H1","H2","H3"]])
    logZ = np.logaddexp.reduce(logvals)
    return {h: float(loglikes[h] - logZ) for h in ["H1","H2","H3"]}