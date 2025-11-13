import time, math
import networkx as nx

# perception returns a dict (type, id, x, y, speed, conf, ts)
def mock_perception():
    return {'type':'vehicle','id':'v1','x':10.0,'y':2.5,'speed':8.5,'conf':0.92,'ts':time.time()}

G = nx.DiGraph()  # event graph

# lift perception to graph node
p = mock_perception()
G.add_node(p['id'], kind=p['type'], x=p['x'], y=p['y'],
           speed=p['speed'], conf=p['conf'], ts=p['ts'], prov='camera_front')

# add static stop-sign node for demo
G.add_node('stop1', kind='stop_sign', x=15.0, y=2.5)
G.add_edge(p['id'],'stop1', relation='approach', dist=5.0)

# simple rule: approach within d and speed > v
def rule_approach_violation(graph, vehicle_id, stop_id, d=6.0, v=5.0):
    if graph.has_edge(vehicle_id, stop_id):
        e = graph[vehicle_id][stop_id]
        if e.get('dist', 999) <= d:
            node = graph.nodes[vehicle_id]
            # compute combined score per Eq.(2) simple form
            log_like = math.log(max(node['conf'],1e-6))
            log_rule = math.log(0.9) if node['speed']>v else math.log(0.1)
            score = 1.0*log_like + 1.0*log_rule
            return {'hyp':'stop_violation','score':score,'evidence':(vehicle_id,stop_id)}
    return None

hyp = rule_approach_violation(G,'v1','stop1')
print(hyp)  # downstream consumer will decide promote/demote