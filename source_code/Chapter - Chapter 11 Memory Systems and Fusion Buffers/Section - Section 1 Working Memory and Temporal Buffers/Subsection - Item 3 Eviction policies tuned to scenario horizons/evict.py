import heapq, time, math, json

class BufferItem:
    def __init__(self, id, created, relevance, entropy, cost, provenance):
        self.id = id
        self.created = created
        self.relevance = relevance      # [0,1]
        self.entropy = entropy          # [0,1]
        self.cost = cost                # bytes
        self.provenance = provenance    # audit weight

def score_item(item, H_s, weights):
    age = time.time() - item.created
    alpha,beta,gamma,delta = weights
    return (alpha * math.exp(-age / H_s)
            + beta * item.relevance
            + gamma * (1.0 - item.entropy)
            + 0.1 * item.provenance
            - delta * (item.cost / 1e6))  # cost normalized

class EvictionManager:
    def __init__(self, budget_bytes, weights):
        self.budget = budget_bytes
        self.weights = weights
        self.store = {}            # id -> BufferItem
        self.current_bytes = 0

    def add(self, item, H_s, demote_fn):
        self.store[item.id] = item
        self.current_bytes += item.cost
        self._enforce_budget(H_s, demote_fn)

    def _enforce_budget(self, H_s, demote_fn):
        if self.current_bytes <= self.budget:
            return
        heap = []
        for it in self.store.values():
            heapq.heappush(heap, (score_item(it, H_s, self.weights), it.id))
        # evict lowest-score until under budget
        while self.current_bytes > self.budget and heap:
            score, evict_id = heapq.heappop(heap)
            item = self.store.pop(evict_id)
            self.current_bytes -= item.cost
            demote_fn(item)                 # archive or compress
            # log rationale for assurance
            print(json.dumps({"action":"evict","id":evict_id,"score":score,"time":time.time()}))