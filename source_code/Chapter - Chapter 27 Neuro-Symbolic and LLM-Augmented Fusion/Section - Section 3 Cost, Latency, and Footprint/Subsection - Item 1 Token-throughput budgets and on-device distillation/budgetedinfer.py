import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load distilled model (small specialist)
MODEL_NAME = "distilroberta-base"  # example; use a distilled LM artifact
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to("cpu")

class BudgetManager:
    def __init__(self, token_max, qps_max, latency_slo):
        self.token_max = token_max
        self.qps_max = qps_max
        self.latency_slo = latency_slo
        self._window_start = time.time()
        self._count = 0

    def admit(self, tokens):
        now = time.time()
        # simple sliding window QPS
        if now - self._window_start > 1.0:
            self._window_start = now; self._count = 0
        if self._count >= self.qps_max or tokens > self.token_max:
            return False
        self._count += 1
        return True

cache = {}  # simple memoization: prompt -> (response, timestamp)
budget = BudgetManager(token_max=256, qps_max=5, latency_slo=0.25)

def infer(prompt):
    # cache hit
    if prompt in cache:
        return cache[prompt][0]
    toks = tokenizer.encode(prompt)
    if not budget.admit(len(toks)):
        raise RuntimeError("Budget exceeded: reject or degrade query")
    start = time.time()
    with torch.no_grad():
        out = model.generate(torch.tensor([toks]), max_new_tokens=64, do_sample=False)
    latency = time.time() - start
    if latency > budget.latency_slo:
        # optionally trigger degraded path or cloud fallback
        pass
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    cache[prompt] = (resp, time.time())
    return resp

# Example usage
print(infer("Summarize recent sensor anomalies in 2 bullets."))