import time, collections

# Minimal rule and rulebase models.
class Production:
    def __init__(self, condition, action, meta):
        self.condition = condition  # callable: WM -> bool
        self.action = action        # callable: WM -> None
        self.meta = meta            # provenance, stats

class RuleBase:
    def __init__(self):
        self.productions = []

    def fire(self, wm):
        # return first matching production (deterministic policy)
        for p in self.productions:
            if p.condition(wm):
                start = time.time()
                p.action(wm)
                p.meta['hits'] += 1
                p.meta['last_hit'] = time.time()
                p.meta['exec_time'] += time.time() - start
                return True
        return False

# Chunk generator: compile when a trace repeats.
class Chunker:
    def __init__(self, rulebase, frequency_threshold=3):
        self.rb = rulebase
        self.trace_counts = collections.Counter()
        self.frequency_threshold = frequency_threshold

    def observe_trace(self, trace_key, trace_payload):
        # trace_key: hashable summary of the working-memory trace
        self.trace_counts[trace_key] += 1
        if self.trace_counts[trace_key] == self.frequency_threshold:
            self.compile_chunk(trace_key, trace_payload)

    def compile_chunk(self, key, payload):
        # simple compile: create a production that matches the key
        t0 = time.time()
        def cond(wm): return wm.get('trace_summary') == key
        def act(wm):
            # action derived from payload; here we annotate WM for demo
            wm['action_taken'] = payload['action']
        meta = {'compiled_at': t0, 'hits': 0, 'exec_time': 0.0}
        prod = Production(cond, act, meta)
        self.rb.productions.append(prod)
        # record compile latency for monitoring
        compile_latency = time.time() - t0
        print(f"Compiled chunk {key} in {compile_latency:.3f}s")
        # attach provenance for assurance
        prod.meta['provenance'] = payload.get('evidence_traces', [])
        prod.meta['confidence'] = payload.get('validation_score', 0.0)

# Demo usage
if __name__ == "__main__":
    rb = RuleBase()
    ch = Chunker(rb, frequency_threshold=2)
    wm = {}
    # simulate repeated traces
    for i in range(4):
        wm['trace_summary'] = 'L1-L2-pattern-A'
        ch.observe_trace('L1-L2-pattern-A', {'action':'promote_hypothesis','validation_score':0.9})
        # fire: chunk will exist after threshold reached
        fired = rb.fire(wm)
        print("Fired:", fired, "WM:", wm)