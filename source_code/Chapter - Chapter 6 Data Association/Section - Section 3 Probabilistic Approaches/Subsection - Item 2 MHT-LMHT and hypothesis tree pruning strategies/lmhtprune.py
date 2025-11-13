import math, heapq, collections

class Hypothesis:
    def __init__(self, label, logw, history):
        self.label = label            # persistent label (string)
        self.logw = logw              # log weight
        self.history = history        # list of (scanIndex, assoc)

# compute child log weight from parent and local log likelihoods
def expand(parentHyps, localLogLikes):
    children = []
    for p in parentHyps:
        for assoc, ll in localLogLikes.items():
            # new history append
            h = Hypothesis(p.label, p.logw + ll, p.history + [assoc])
            children.append(h)
    return children

def pruneGlobal(hypotheses, topK, nScanCommit, currentScan):
    # keep top-K by logw
    bestK = heapq.nlargest(topK, hypotheses, key=lambda h: h.logw)
    # N-scan commit: remove histories that differ from best beyond window
    committed = []
    windowStart = currentScan - nScanCommit
    best = bestK[0]
    for h in bestK:
        # compare history before windowStart; simple prefix compare
        if all(a==b for a,b in zip(h.history[:windowStart+1], best.history[:windowStart+1])):
            committed.append(h)
    return committed or bestK  # return committed set or fallback