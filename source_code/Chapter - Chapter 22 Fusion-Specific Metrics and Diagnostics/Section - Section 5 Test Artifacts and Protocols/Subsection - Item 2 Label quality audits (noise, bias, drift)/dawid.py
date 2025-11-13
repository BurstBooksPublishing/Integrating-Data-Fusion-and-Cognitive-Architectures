import numpy as np
import pandas as pd

def dawid_skene(labels_df, classes, max_iter=50, tol=1e-6):
    # labels_df: rows=item_id, columns=annotator_id (NaN if unlabeled)
    items = labels_df.index.values
    annotators = labels_df.columns.values
    K = len(classes)
    N = len(items)

    # init priors uniformly
    pi = np.full(K, 1.0/K)
    # init annotator confusion matrices with small smoothing
    theta = {a: np.eye(K) * 0.9 + 0.1/K for a in annotators}

    # map class -> index
    cls2i = {c:i for i,c in enumerate(classes)}

    for it in range(max_iter):
        # E-step: posterior weights w_{i,k}
        W = np.zeros((N, K))
        for i_idx,i in enumerate(items):
            probs = pi.copy()
            for a in annotators:
                lab = labels_df.at[i,a]
                if pd.isna(lab): continue
                lidx = cls2i[lab]
                probs *= theta[a][:, lidx]
            probs_sum = probs.sum()
            if probs_sum == 0:
                probs = np.full(K, 1.0/K)
                probs_sum = 1.0
            W[i_idx,:] = probs / probs_sum

        # M-step: update pi and theta
        pi_new = W.sum(axis=0) / N
        theta_new = {}
        for a in annotators:
            num = np.zeros((K,K))
            denom = np.zeros(K)
            for i_idx,i in enumerate(items):
                lab = labels_df.at[i,a]
                if pd.isna(lab): continue
                lidx = cls2i[lab]
                num[:, lidx] += W[i_idx,:]
                denom += W[i_idx,:]
            # avoid division by zero
            denom = np.where(denom==0, 1e-8, denom)
            theta_new[a] = (num / denom[:,None])
        # check convergence
        if np.linalg.norm(pi_new - pi) < tol:
            pi = pi_new; theta = theta_new; break
        pi = pi_new; theta = theta_new
    return pi, theta, W  # priors, annotator matrices, posterior per item