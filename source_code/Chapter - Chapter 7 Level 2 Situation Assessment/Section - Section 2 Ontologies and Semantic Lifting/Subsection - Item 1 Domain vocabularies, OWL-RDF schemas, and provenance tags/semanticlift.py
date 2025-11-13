from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
import datetime, math

# Namespaces
EX = Namespace("http://example.org/ns/")
PROV = Namespace("http://www.w3.org/ns/prov#")

g = Graph()
g.bind("ex", EX); g.bind("prov", PROV)

# Inputs: two sensors with confidences
track_uri = EX["track-42"]
entity_uri = EX["Vehicle-123"]
p_sensor = [0.72, 0.85]

# 1) create entity and observation relation
g.add((track_uri, RDF.type, EX.Track))
g.add((track_uri, EX.observedEntity, entity_uri))
g.add((entity_uri, RDF.type, EX.Vehicle))

# 2) aggregate confidences per Eq. (1)
p_agg = 1 - math.prod([1 - p for p in p_sensor])
g.add((track_uri, EX.confidence, Literal(p_agg, datatype=XSD.float)))

# 3) provenance: generation activity with timestamp and sources
activity = EX["lift-activity-1"]
g.add((activity, RDF.type, PROV.Activity))
g.add((activity, PROV.startedAtTime, Literal(datetime.datetime.utcnow().isoformat(), datatype=XSD.dateTime)))
g.add((activity, PROV.used, EX["radar-1"])); g.add((activity, PROV.used, EX["lidar-2"]))
g.add((track_uri, PROV.wasGeneratedBy, activity))

# Output TTL for interop
print(g.serialize(format="turtle").decode())