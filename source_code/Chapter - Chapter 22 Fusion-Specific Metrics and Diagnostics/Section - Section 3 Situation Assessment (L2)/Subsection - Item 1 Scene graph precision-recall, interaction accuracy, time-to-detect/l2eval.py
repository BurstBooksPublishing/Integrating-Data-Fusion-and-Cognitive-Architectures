import networkx as nx, numpy as np
# gt_graphs, pred_graphs: dict[time] -> nx.DiGraph with node attr 'class' and optional 'track_id'
# interactions: list of (start, end, subj_gt, obj_gt, rel) for ground truth; pred_interactions similar.

def node_matches(gt_node, pred_node):
    return gt_node.get('class') == pred_node.get('class') and gt_node.get('track_id')==pred_node.get('track_id')

def edge_matches(gt_edge_attrs, pred_edge_attrs):
    return gt_edge_attrs.get('relation') == pred_edge_attrs.get('relation')

def compute_node_edge_pr(gt_graph, pred_graph):
    # align by track_id where available
    gt_nodes = {n:gt_graph.nodes[n] for n in gt_graph.nodes}
    pred_nodes = {n:pred_graph.nodes[n] for n in pred_graph.nodes}
    tp_n = sum(any(node_matches(gt_nodes[g], pred_nodes[p]) for p in pred_nodes) for g in gt_nodes)
    fp_n = len(pred_nodes) - tp_n
    fn_n = len(gt_nodes) - tp_n
    # edges
    tp_e = 0
    for u,v in pred_graph.edges:
        for gu,gv in gt_graph.edges:
            if pred_graph.nodes[u].get('track_id')==gt_graph.nodes[gu].get('track_id') and \
               pred_graph.nodes[v].get('track_id')==gt_graph.nodes[gv].get('track_id') and \
               edge_matches(gt_graph.edges[gu,gv], pred_graph.edges[u,v]):
                tp_e += 1; break
    fp_e = pred_graph.number_of_edges() - tp_e
    fn_e = gt_graph.number_of_edges() - tp_e
    return tp_n,fp_n,fn_n,tp_e,fp_e,fn_e

def ttd_for_events(gt_events, pred_events):
    # match by relation+participants; compute latency to first matched detection
    latencies=[]
    for ge in gt_events:
        start,_,s,o,r = ge
        # find earliest pred event with same participants+rel
        cand = [p for p in pred_events if p[3]==r and p[1]==o and p[2]==s] # ordering depends on schema
        if not cand: continue
        detect_time = min(c[0] for c in cand)
        latencies.append(max(0, detect_time-start))
    return np.mean(latencies) if latencies else np.nan