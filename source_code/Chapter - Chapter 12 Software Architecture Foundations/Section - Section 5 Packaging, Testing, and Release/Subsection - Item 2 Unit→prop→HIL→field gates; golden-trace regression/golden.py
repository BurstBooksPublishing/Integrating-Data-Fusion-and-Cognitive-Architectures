#!/usr/bin/env python3
import json,subprocess,sys,csv,math
import numpy as np

# load golden trace (timestamps, inputs, ref_states, ref_covs)
with open('golden_trace.json') as f:
    trace = json.load(f)

# run node under test; example: docker run --rm mynode:latest
proc = subprocess.Popen(['docker','run','--rm','-i','mynode:latest'],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

# feed inputs and collect outputs (protocol depends on node)
outputs = []
for step in trace['steps']:
    proc.stdin.write(json.dumps(step['input']) + "\n")
    proc.stdin.flush()
    out_line = proc.stdout.readline()
    outputs.append(json.loads(out_line))  # node emits JSON states

proc.terminate()

# compute Mahalanobis distances and report
d2 = []
for i,step in enumerate(trace['steps']):
    x_ref = np.array(step['ref_state'])
    P = np.array(step['ref_cov'])
    x_new = np.array(outputs[i]['state'])
    delta = x_ref - x_new
    try:
        invP = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        print("Covariance singular at step", i); sys.exit(2)
    d2.append(float(delta.T @ invP @ delta))

# summarize and decide
chi2_thresh = 9.21  # example for n=2, alpha=0.01
pass_frac = sum(1 for v in d2 if v <= chi2_thresh) / len(d2)
print(f"pass_frac={pass_frac:.3f}")
with open('golden_report.csv','w') as csvf:
    w = csv.writer(csvf); w.writerow(['t','d2']); 
    w.writerows([(i,float(v)) for i,v in enumerate(d2)])

if pass_frac < 0.9:  # gate threshold
    sys.exit(1)
print("GOLDEN TRACE REGRESSION PASSED")