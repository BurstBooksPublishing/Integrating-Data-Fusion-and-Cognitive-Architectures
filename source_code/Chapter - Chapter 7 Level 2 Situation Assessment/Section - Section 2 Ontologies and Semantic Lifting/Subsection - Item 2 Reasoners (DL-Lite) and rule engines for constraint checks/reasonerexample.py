from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD
from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate

EX = Namespace("http://example.org/")
g = Graph()
g.bind("ex", EX)

# ABox: a track with speed and provenance triple
track = EX["track123"]
g.add((track, RDF.type, EX.Track))
g.add((track, EX.hasSpeed, Literal(22.5, datatype=XSD.double)))
g.add((track, EX.provenance, EX.sensorA))

# TBox: Event superclass (simple example)
g.add((EX.SpeedingEvent, RDF.type, EX.EventClass))  # placeholder

# Run OWL RL materialization (adds inferred triples)
DeductiveClosure(OWLRL_Semantics).expand(g)

# Closed-world operational rule via SPARQL UPDATE (threshold check)
# Insert an Event individual when speed > 20.0
update = """
PREFIX ex: 
INSERT {
  ?t ex:hasEvent ?ev .
  ?ev a ex:SpeedingEvent ;
      ex:derivedFrom ?t .
}
WHERE {
  ?t a ex:Track ;
     ex:hasSpeed ?s .
  FILTER(xsd:double(?s) > 20.0)
  BIND(IRI(CONCAT(str(?t), "_evt")) AS ?ev)
}
"""
g.update(update)

# SHACL: ensure every Event has provenance (simple shape as TTL string)
shacl_ttl = """
@prefix sh:  .
@prefix ex:  .
ex:EventShape
  a sh:NodeShape ;
  sh:targetClass ex:SpeedingEvent ;
  sh:property [ sh:path ex:derivedFrom ; sh:minCount 1 ] ;
  sh:property [ sh:path ex:derivedFrom/ex:provenance ; sh:minCount 1 ] .
"""
# Validate graph (returns conforms, results graph, text)
conforms, results_graph, results_text = validate(g, shacl_graph=shacl_ttl, data_graph=g, inference='rdfs', debug=False)
print("SHACL conforms:", conforms)
if not conforms:
    print(results_text)  # brief inline report