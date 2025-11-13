from rdflib import Graph
import json, math

# load ontologies (files: old.owl, new.owl)
g_old = Graph().parse("old.owl", format="xml")
g_new = Graph().parse("new.owl", format="xml")

# structural similarity: Jaccard over triple sets
triples_old = set(g_old)
triples_new = set(g_new)
inter = triples_old & triples_new
union = triples_old | triples_new
S_struct = len(inter) / len(union) if union else 1.0

# simple logical check: count violated constraints via SPARQL sample
# example: ensure no instances of deprecated property ex:forbiddenProp
q_viol = """
PREFIX ex: 
SELECT (COUNT(*) AS ?viol) WHERE { ?s ex:forbiddenProp ?o . }
"""
viol_old = int(list(g_old.query(q_viol))[0][0]) if list(g_old.query(q_viol)) else 0
viol_new = int(list(g_new.query(q_viol))[0][0]) if list(g_new.query(q_viol)) else 0
V_logic = max(0, viol_new - viol_old)

# downstream task check: map sample labels through both ontologies
# assume mapping function that yields labels.json for each ontology
def run_labeler(ontology_file, samples_file="samples.json"):
    # placeholder: call to model/service that uses ontology
    # here we return labels loaded from synthetic file per ontology
    fname = ontology_file.replace(".owl", ".labels.json")
    with open(fname) as f: return json.load(f)

labels_old = run_labeler("old.owl")
labels_new = run_labeler("new.owl")

# compute drop in accuracy on golden set
with open("golden_labels.json") as f: golden = json.load(f)
def acc(pred, true):
    return sum(1 for k in true if pred.get(k)==true[k]) / len(true)
acc_old = acc(labels_old, golden)
acc_new = acc(labels_new, golden)
D_task = max(0.0, acc_old - acc_new)

# regression score as in equation (weights chosen by policy)
alpha, beta, gamma = 1.0, 2.0, 5.0
R = alpha*(1 - S_struct) + beta*V_logic + gamma*D_task

print(f"S_struct={S_struct:.3f}, V_logic={V_logic}, D_task={D_task:.3f}, R={R:.3f}")

# decision rule: block if above policy threshold
THRESHOLD = 0.5
if R > THRESHOLD:
    raise SystemExit("Block: regression detected; require manual review.")
else:
    print("Pass: safe to canary rollout.")