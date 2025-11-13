import pandas as pd
import networkx as nx
from shapely.geometry import Point, shape
# tracks: DataFrame with ['id','x','y','ts','speed']
# zones: GeoJSON-like list of {'id','geom','type','emission_limit'}
def build_situation_graph(tracks, zones, window_start, window_end):
    G = nx.DiGraph()
    # add zone nodes with semantics
    for z in zones:
        G.add_node(z['id'], type=z['type'], geom=z['geom'],
                   emission_limit=z.get('emission_limit', float('inf')),
                   vehicle_count=0, est_emissions=0.0)
    # map tracks to zones and aggregate flows
    recent = tracks[(tracks.ts >= window_start) & (tracks.ts < window_end)]
    # simple origin-destination based on first/last location in window
    od = {}
    for tid, g in recent.groupby('id'):
        pts = [Point(xy) for xy in zip(g.x, g.y)]
        zones_hit = [z['id'] for z in zones if any(shape(z['geom']).contains(p) for p in pts)]
        if len(zones_hit) >= 2:
            o, d = zones_hit[0], zones_hit[-1]
            od[(o,d)] = od.get((o,d), 0) + 1
    # add edges with flow counts
    for (o,d), cnt in od.items():
        G.add_edge(o, d, vehicle_flow=cnt)
        G.nodes[o]['vehicle_count'] += cnt
        # crude emissions: assume per-vehicle emission factor
        G.nodes[o]['est_emissions'] += cnt * 0.2
    # hazard detection: compare estimated emissions to limits
    for nid, data in G.nodes(data=True):
        if data['est_emissions'] > data['emission_limit']:
            data['hazard'] = True
        else:
            data['hazard'] = False
    return G
# example usage omitted for brevity