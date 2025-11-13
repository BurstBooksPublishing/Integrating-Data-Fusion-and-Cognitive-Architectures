import numpy as np
from heapq import heappush, heappop

# Grid shortest path (Manhattan cost), deterministic agent assumed.
def astar(start, goal, grid_size):
    # simple BFS/A* using Manhattan heuristic
    openq = [(0, start)]
    came = {start: None}
    cost = {start: 0}
    while openq:
        _, cur = heappop(openq)
        if cur == goal: break
        x,y = cur
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nb = (x+dx, y+dy)
            if 0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size:
                nc = cost[cur] + 1
                if nb not in cost or nc < cost[nb]:
                    cost[nb] = nc
                    came[nb] = cur
                    priority = nc + abs(goal[0]-nb[0])+abs(goal[1]-nb[1])
                    heappush(openq, (priority, nb))
    # reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node); node = came.get(node)
    return list(reversed(path))

# likelihood of noisy observations under a planned trajectory
def traj_likelihood(obs, path, sigma=0.8):
    # obs: list of (x,y) floats; path: list of integer grid positions
    # align by time, allow path shorter: repeat final state
    L = 1.0
    var = sigma**2
    for t,o in enumerate(obs):
        s = path[min(t, len(path)-1)]
        diff = np.array(o) - np.array(s)
        L *= np.exp(-0.5*(diff@diff)/var)/(2*np.pi*var)**0.5
    return L

# define scenario
grid = 10
start = (1,1)
goals = [(8,8),(1,8),(8,1)]
priors = np.array([0.5, 0.25, 0.25])   # contextual prior

# simulate true goal and noisy observations
true_goal = goals[0]
true_path = astar(start,true_goal,grid)
np.random.seed(2)
obs = [np.array(p) + np.random.normal(0,0.8,2) for p in true_path[:8]] # partial trace

# compute posterior
likes = np.array([traj_likelihood(obs, astar(start,g,grid)) for g in goals])
unnorm = priors * likes
post = unnorm / unnorm.sum()
print("Posteriors:", post)                  # actionable intent probabilities
print("Entropy:", -np.sum(post*np.log(post+1e-12)))  # diagnostic