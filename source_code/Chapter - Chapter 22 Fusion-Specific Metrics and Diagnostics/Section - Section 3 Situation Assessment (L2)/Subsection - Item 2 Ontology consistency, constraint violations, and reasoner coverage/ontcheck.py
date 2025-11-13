from owlready2 import get_ontology, sync_reasoner_pellet  # requires Pellet/JDK
from pyshacl import validate
import rdflib, time

# load ontology and instance data (T)
onto = get_ontology("file://./domain.owl").load()
g = rdflib.Graph()
g.parse("instances.ttl", format="turtle")

# SHACL shapes for closed-world invariants
shapes = rdflib.Graph()
shapes.parse("shapes.ttl", format="turtle")

# SHACL validation (fast, closed-world rules)
conforms, v_graph, v_text = validate(g, shacl_graph=shapes, inference="rdfs", debug=False)

# DL reasoning for global consistency (may be slow)
start = time.time()
try:
    sync_reasoner_pellet([onto], infer_property_values=True)  # may raise if timeout
    consistent = True
except Exception as e:
    consistent = False
reason_time = time.time() - start

# simple coverage proxy: SHACL rules checked + DL run success
total_axioms = 100  # pre-counted design-time
checked = (len(list(shapes))+ (1 if consistent else 0))
coverage = checked / total_axioms

print(f"SHACL conforms: {conforms}, DL consistent: {consistent}, time: {reason_time:.2f}s")
print(f"coverage≈{coverage:.2%}")
# persist violation triples and reasoner logs for audit
v_graph.serialize("violations.ttl", format="turtle")