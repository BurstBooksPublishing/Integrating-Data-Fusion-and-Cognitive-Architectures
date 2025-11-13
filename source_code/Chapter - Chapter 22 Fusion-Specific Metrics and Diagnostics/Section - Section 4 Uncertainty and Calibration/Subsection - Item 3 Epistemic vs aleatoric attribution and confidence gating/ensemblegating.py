import numpy as np
# preds: [n_models, n_samples, n_dims]; vars: [n_models, n_samples, n_dims] optional
def decompose_uncertainty(preds, vars_pred=None):
    # epistemic: variance across model means
    mean_per_model = preds.mean(axis=1)                # [n_models, n_dims]
    epistemic = mean_per_model.var(axis=0, ddof=0)     # [n_dims]
    # aleatoric: average predicted variance, or residual within-model variance
    if vars_pred is not None:
        aleatoric = vars_pred.mean(axis=0).mean(axis=0)  # [n_dims]
    else:
        # fallback: expected within-model variance from sample variance minus epistemic
        total = preds.reshape(-1, preds.shape[-1]).var(axis=0)
        aleatoric = np.maximum(total - epistemic, 1e-6)
    return aleatoric, epistemic

# gating decision (scalar thresholds for brevity)
def gate_decision(alea, epi, T_alea=0.5, T_epi=1.0):
    if epi > T_epi:
        return "ABSTAIN"    # high epistemic => human/cautious fallback
    if alea > T_alea:
        return "LOW_CONFIDENCE"  # soft degrade e.g., request additional sensors
    return "ACCEPT"