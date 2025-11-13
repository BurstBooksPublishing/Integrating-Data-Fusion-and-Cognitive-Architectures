#!/usr/bin/env python3
import asyncio, time, copy, hashlib, json

# Simple state representing L0-L3 artifacts
initial_state = {"frame":0, "tracks":[], "situation":None, "provenance":[]}

def sign(state):
    h = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return {"state": state, "sig": h, "ts": time.time()}

class RollbackNode:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.children = []

class RollbackTree:
    def __init__(self):
        self.root = None
    def add_root(self, ckpt):
        self.root = RollbackNode(ckpt)
        return self.root
    def branch(self, node, ckpt):
        child = RollbackNode(ckpt)
        node.children.append(child)
        return child

async def fuse_loop(state, tree_root):
    node = tree_root
    try:
        for t in range(1,6):
            # simulate processing
            state["frame"] = t
            state["tracks"].append({"id":t, "pos":(t, t*0.5)})
            ck = sign(copy.deepcopy(state))
            node = tree.branch(node, ck)  # checkpoint at decision point
            await asyncio.sleep(0.1)
            # simulate jam at t==3
            if t==3:
                raise RuntimeError("simulated jammer detected")
    except Exception as e:
        print("Fault:", e)
        # trigger soft kill: stop actuation, preserve data
        print("Invoking kill-switch: soft")
        # pick recovery node (parent) and rollback
        recovery = node  # pick most recent safe checkpoint
        state = copy.deepcopy(recovery.checkpoint["state"])
        print("Rolled back to frame", state["frame"])
    return state

if __name__ == "__main__":
    tree = RollbackTree()
    tree.add_root(sign(initial_state))
    state = copy.deepcopy(initial_state)
    state = asyncio.run(fuse_loop(state, tree))
    print("Final state:", state)