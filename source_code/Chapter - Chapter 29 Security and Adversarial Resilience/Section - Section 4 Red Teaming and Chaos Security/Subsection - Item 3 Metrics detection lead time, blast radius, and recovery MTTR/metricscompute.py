import pandas as pd
import networkx as nx
from datetime import timedelta

# load events: time,event_type,component,incident_id,extra
events = pd.read_csv("events.csv", parse_dates=["time"])

# build system graph with importance weight attribute
G = nx.read_gpickle("system_graph.gpk")  # nodes have 'weight' attr

def detection_lead_time(df):
    # assumes 'attack_start' and 'detect' events exist per incident
    grp = df.groupby("incident_id")
    dl = grp.apply(lambda g: (g.loc[g.event_type=="detect","time"].min()
                              - g.loc[g.event_type=="attack_start","time"].min()).total_seconds())
    return dl.dropna()

def blast_radius(df):
    # components affected are those with 'affected' event types
    affected = df[df.event_type=="affected"].groupby("incident_id")["component"].unique()
    br = {}
    total = sum(nx.get_node_attributes(G,"weight").values())
    for inc, comps in affected.items():
        w = sum(G.nodes[c]["weight"] for c in comps if c in G)
        br[inc] = w/total
    return pd.Series(br)

def mttr(df):
    grp = df.groupby("incident_id")
    def recovery_time(g):
        t_detect = g.loc[g.event_type=="detect","time"].min()
        t_recover = g.loc[g.event_type=="recovered","time"].max()
        return (t_recover - t_detect).total_seconds() if pd.notna(t_detect) and pd.notna(t_recover) else None
    return grp.apply(recovery_time).dropna()

# compute metrics
dl = detection_lead_time(events)
br = blast_radius(events)
mt = mttr(events)

# aggregate summary
summary = pd.DataFrame({
    "DL_median_s": dl.median(),
    "DL_p95_s": dl.quantile(0.95),
    "BR_median": br.median(),
    "MTTR_p95_s": mt.quantile(0.95)
}, index=[0])
summary.to_csv("metrics_summary.csv", index=False)