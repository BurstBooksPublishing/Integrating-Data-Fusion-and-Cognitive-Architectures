# simple multi-resource bin-packing and autoscale decision
from typing import List, Dict, Tuple

# node and task models
Node = Dict[str,int]   # e.g. {'cpu': 1600, 'mem': 32768, 'gpu': 1}
Task = Dict[str,int]   # e.g. {'cpu': 200, 'mem': 2048, 'gpu': 0}

def fits(node:Node, used:Node, task:Task) -> bool:
    return all(used[r]+task.get(r,0) <= node[r] for r in node)

def pack_tasks(tasks:List[Task], nodes:List[Node]) -> Tuple[Dict[int,List[int]], List[int]]:
    # sort tasks by descending CPU+mem score
    tasks_idx = sorted(range(len(tasks)), key=lambda i: (tasks[i]['cpu']*1.0 + tasks[i]['mem']/1024), reverse=True)
    used = [ {r:0 for r in node} for node in nodes ]
    assignment = {j:[] for j in range(len(nodes))}
    activate = [0]*len(nodes)  # 1 means suggested to activate
    for i in tasks_idx:
        t = tasks[i]
        placed=False
        # try active nodes first
        for j,node in enumerate(nodes):
            if fits(node, used[j], t):
                used[j] = {r: used[j][r]+t.get(r,0) for r in node}
                assignment[j].append(i)
                activate[j]=1
                placed=True
                break
        if not placed:
            # suggest spinning new node (scale-out)
            for j,node in enumerate(nodes):
                if activate[j]==0 and fits(node, used[j], t):
                    used[j] = {r: used[j][r]+t.get(r,0) for r in node}
                    assignment[j].append(i)
                    activate[j]=1
                    placed=True
                    break
        if not placed:
            # cannot place task: signal scale-up requirement (e.g., larger instance or add node)
            assignment.setdefault(-1,[]).append(i)  # -1 = unplaced
    return assignment, activate

# small demo: tasks and node pool
tasks = [{'cpu':300,'mem':4096,'gpu':0},{'cpu':800,'mem':8192,'gpu':1},{'cpu':150,'mem':1024,'gpu':0}]
nodes = [{'cpu':1600,'mem':32768,'gpu':1},{'cpu':800,'mem':8192,'gpu':0}]
print(pack_tasks(tasks,nodes))  # prints placement and node-activation suggestions