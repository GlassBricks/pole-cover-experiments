import random
from collections import defaultdict
from heapq import heappop, heappush
from math import ceil
from typing import Callable

import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult

from pole_cover import CandidatePole, solve_approximate_pole_cover, euclid_dist_squared, manhattan_dist, get_center, \
    euclid_dist
from pole_graph import AdjList, Blocker, Pole, PoleGraph
from set_cover import Coverage, solve_set_cover, Pos


def fill_random(
        graph: PoleGraph,
        blocker_density: float,
        powerable_density: float,
        size: int,
        center: Pos = (0, 0)
):
    all_pts = {(x, y) for x in range(size) for y in range(size)}
    num_pts = size * size
    blockers = random.sample(list(all_pts), ceil(num_pts * blocker_density))
    all_pts.difference_update(blockers)
    powerables = random.sample(list(all_pts), ceil(num_pts * powerable_density))

    for x, y in blockers:
        pt2 = center[0] + x - size // 2, center[1] + y - size // 2
        graph.nodes[pt2] = Blocker
    for pt in powerables:
        pt2 = center[0] + pt[0] - size // 2, center[1] + pt[1] - size // 2
        graph.nodes[pt2] = pt2

    return graph


# use euclidean distance as the edge weight
def mst(adj_list: AdjList, start: Pos = None) -> AdjList:
    pts = set(adj_list.keys())
    res = defaultdict(set)
    start = min(pts, key=lambda pt: euclid_dist_squared(pt, start)) if start else next(iter(pts))
    visited = set()
    heap = [(0, start, start)]
    while heap:
        weight, node, parent = heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node != parent:
            res[parent].add(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                heappush(heap, ((node[0] - neighbor[0]) ** 2 + (node[1] - neighbor[1]) ** 2, neighbor, node))
    return res


def plot_adj_list(adj_list: AdjList):
    for pt, neighbors in adj_list.items():
        # directed
        for neigh in neighbors:
            x1, y1 = pt
            x2, y2 = neigh
            dx = x2 - x1
            dy = y2 - y1
            plt.arrow(
                x1, y1, dx, dy,
                head_width=0.3, length_includes_head=True,
                alpha=0.6, zorder=10,
            )
            # plt.plot([x1, x2], [y1, y2], 'k-')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()


g = PoleGraph(7.5, 5)
fill_random(g, 0.5, 0.15, 10, (-20, -20))
fill_random(g, 0.5, 0.15, 10, (20, -20))
fill_random(g, 0.5, 0.15, 10, (20, 20))
fill_random(g, 0.5, 0.15, 10, (-20, 20))
g.nodes[(0, 0)] = str((0, 0))

cand_poles = g.all_candidate_poles()
center = get_center(cand_poles)
for pole in cand_poles:
    pole.cost = euclid_dist(pole.pos, center) / 1000 + 1


def try_with_solver(
        name: str,
        solver: Callable[[list[CandidatePole]], tuple[list[Coverage], OptimizeResult]],
):
    poles, res, center_pos = solver(cand_poles)

    # ugly, but works for now
    for pole in poles:
        g.nodes[pole.pos] = Pole
    pole_adj = g.pole_adj_list()

    g.matplot(show=False)
    plt.title(name)
    plt.suptitle(f"Num poles: {round(sum(res.x))}")
    plot_adj_list(mst(pole_adj, center_pos))
    plt.show()

    for pole in poles:
        del g.nodes[pole.pos]


def dist_fun(p1: CandidatePole, p2: CandidatePole) -> float:
    return g.pole_reach * 2 + manhattan_dist(p1.pos, p2.pos)


# center, dists = get_pole_dists(cand_poles, dist_fun)
# lt = get_lt(center, dists)
# adjlist: AdjList = {}
# pos_to_pole = {pole.pos: pole for pole in cand_poles}
# for pole in cand_poles:
#     adjlist[pole.pos] = {
#         n for n in pole.pole_neighbors
#         if n in pos_to_pole and lt(pole, pos_to_pole[n])
#     }
# plot_adj_list(adjlist)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

try_with_solver(
    "set_cover",
    lambda coverages: solve_set_cover(coverages)
)

try_with_solver(
    "approx_pole_cover",
    lambda coverages: solve_approximate_pole_cover(coverages, dist_fun, milp_options={
        "disp": True,
        "mip_rel_gap": 0.005,
        "time_limit": 300,
    }))
