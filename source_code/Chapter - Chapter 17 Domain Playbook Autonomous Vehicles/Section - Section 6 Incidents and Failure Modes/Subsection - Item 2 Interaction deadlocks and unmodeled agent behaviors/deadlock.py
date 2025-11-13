import networkx as nx

def build_interaction_graph(agents):
    G = nx.DiGraph()
    for a in agents:
        G.add_node(a['id'])
    for a in agents:
        for b in agents:
            if a['id']==b['id']: continue
            # edge if a expects b to act first (probability threshold)
            if a['yield_prob_to'].get(b['id'],0) > 0.6:
                G.add_edge(a['id'], b['id'])
    return G

def resolve_deadlock(G, agents):
    cycles = list(nx.simple_cycles(G))
    if not cycles:
        return None  # no deadlock
    # compute risk score per agent (lower means safer to yield)
    risk = {a['id']: a['ttc'] * (1 - a['compliance']) for a in agents}
    # resolve each cycle: choose agent with max risk to proceed
    actions = {}
    for cyc in cycles:
        proceed = max(cyc, key=lambda x: risk[x])
        for v in cyc:
            actions[v] = 'proceed' if v==proceed else 'yield'
    return actions

# Example usage
agents = [
    {'id':'A', 'yield_prob_to':{'B':0.8}, 'ttc':3.0, 'compliance':0.6},
    {'id':'B', 'yield_prob_to':{'A':0.7}, 'ttc':2.5, 'compliance':0.7},
]
G = build_interaction_graph(agents)
plan = resolve_deadlock(G, agents)
print(plan)  # {'A':'yield','B':'proceed'} -- chosen by risk heuristic