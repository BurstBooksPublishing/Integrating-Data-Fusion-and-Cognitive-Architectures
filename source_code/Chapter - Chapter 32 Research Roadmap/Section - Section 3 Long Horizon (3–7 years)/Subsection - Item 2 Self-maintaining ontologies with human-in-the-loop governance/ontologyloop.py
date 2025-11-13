from rdflib import Graph, URIRef, Literal, Namespace
import json, datetime

# load baseline ontology
g = Graph()
g.parse("baseline_ontology.ttl", format="turtle")  # baseline TTL file

EX = Namespace("http://example.org/")
g.bind("ex", EX)

# candidate triples from fusion/evidence
candidates = [
    (EX.VehicleX, EX.hasBehavior, Literal("evasive_maneuver")),
    (EX.VehicleX, EX.type, EX.UnregisteredVehicle),
]

# simple novelty and conflict checks
def novelty_score(g, candidates):
    new = sum(1 for s,p,o in candidates if (s,p,o) not in g)
    return new / max(len(candidates), 1)

# placeholder conflict rule: disallow assigning 'type' if subject has 'registered' flag
def conflict_score(g, candidates):
    conflicts = 0
    for s,p,o in candidates:
        if p == EX.type and (s, EX.registered, Literal(True)) in g:
            conflicts += 1
    return conflicts / max(len(candidates), 1)

nov = novelty_score(g, candidates)
conf = conflict_score(g, candidates)
w_n, w_c, tau = 0.7, 0.3, 0.5
risk = w_n*nov + w_c*conf

if risk > tau:
    # create human review ticket (simple JSON file)
    ticket = {"id": f"tkt-{datetime.datetime.utcnow().isoformat()}",
              "risk": risk, "novelty": nov, "conflict": conf,
              "candidates": [(str(s), str(p), str(o)) for s,p,o in candidates]}
    with open("review_ticket.json","w") as f:
        json.dump(ticket, f, indent=2)
    print("Human review required:", ticket["id"])
else:
    # auto-apply low-risk changes with provenance
    for s,p,o in candidates:
        g.add((s,p,o))
    prov = (EX.System, EX.lastUpdate, Literal(datetime.datetime.utcnow().isoformat()))
    g.add(prov)
    g.serialize("updated_ontology.ttl", format="turtle")
    print("Applied candidates; saved updated_ontology.ttl")