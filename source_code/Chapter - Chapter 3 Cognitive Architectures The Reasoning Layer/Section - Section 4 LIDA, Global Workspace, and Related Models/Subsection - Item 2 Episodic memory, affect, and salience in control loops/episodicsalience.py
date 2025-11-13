import time, math
import numpy as np

# Simple episodic store (in-memory) and deterministic logger
class EpisodicMemory:
    def __init__(self, decay=0.999):
        self.episodes = []  # list of dicts
        self.decay = decay
    def add(self, ctx, vec, action, outcome, affect):
        self.episodes.append({'t':time.time(), 'ctx':ctx, 'vec':np.array(vec),
                              'action':action, 'outcome':outcome, 'affect':affect})
    def recent_embeddings(self, window=100):
        return np.stack([e['vec'] for e in self.episodes[-window:]]) if self.episodes else np.empty((0,))

class AffectModel:
    def __init__(self, value=0.0, lr=0.1):
        self.value = value
        self.lr = lr
    def update(self, reward, expected):
        td = reward - expected
        self.value += self.lr * td  # conservative TD update

def salience_score(novelty, relevance, affect, weights=(0.4,0.4,0.2)):
    a,b,c = weights
    return a*novelty + b*relevance + c*affect

# Example flow
mem = EpisodicMemory()
aff = AffectModel()

# insert a negative outcome episode (spoofing)
mem.add(ctx='wpA', vec=[0.1,0.9], action='verify', outcome=-1.0, affect=-0.8)
aff.update(reward=-1.0, expected=0.0)  # update affect model

# incoming observation embedding
obs_vec = np.array([0.12,0.88])
# novelty as min distance to recent episodes
embs = mem.recent_embeddings()
novelty = 0.0 if embs.size==0 else np.min(np.linalg.norm(embs - obs_vec, axis=1))
# relevance via simple cosine to task vector
task_vec = np.array([0.0,1.0])
relevance = float(np.dot(obs_vec, task_vec) / (np.linalg.norm(obs_vec)*np.linalg.norm(task_vec)))
# compute salience and gate attention weight
S = salience_score(novelty=1.0/(1.0+novelty), relevance=relevance, affect=aff.value)
attention_weight = min(1.0, max(0.0, 0.1 + 0.9 * S))  # gated attention
print(f"salience={S:.3f}, attention_weight={attention_weight:.3f}")