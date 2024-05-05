import random
from collections import defaultdict
from copy import deepcopy
from heapq import heappop, heappush
from math import ceil
from typing import Callable, NamedTuple, Any

import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult

from pole_cover import solve_approximate_pole_cover, euclid_dist_squared, manhattan_dist, get_center, \
    euclid_dist, get_pole_dists
from pole_graph import PoleGrid, Pos, NonPole, Entity, Pole, CandidatePole, small_pole
from set_cover import solve_set_cover

AdjList = dict[Any, set[Any]]


class QE(NamedTuple):
    d: float
    a: Entity
    b: Entity

    def __lt__(self, other):
        return self.d < other.d

    def __iter__(self):
        return iter((self.d, self.a, self.b))


# use euclidean distance as the edge weight
def mst(adj_list: AdjList) -> AdjList:
    pts = set(adj_list.keys())
    res = defaultdict(set)
    start = next(iter(pts))
    visited = set()
    heap = [QE(0, start, start)]
    while heap:
        weight, node, parent = heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node != parent:
            res[parent].add(node)
        p1 = node.pos
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                p2 = neighbor.pos
                heappush(heap, QE(euclid_dist_squared(p1, p2), neighbor, node))
    return res


def plot_adj_list(adj_list: AdjList):
    for pt, neighbors in adj_list.items():
        # directed
        for neigh in neighbors:
            x1, y1 = pt.pos
            x2, y2 = neigh.pos
            dx = x2 - x1
            dy = y2 - y1
            plt.arrow(
                x1, y1, dx, dy,
                head_width=0.5, length_includes_head=True,
                alpha=0.6, zorder=10,
            )


def fill_random(
        grid: PoleGrid,
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
        grid.add(NonPole.at_tile(pt2, False))

    for pt in powerables:
        pt2 = center[0] + pt[0] - size // 2, center[1] + pt[1] - size // 2
        grid.add(NonPole.at_tile(pt2, True))

    return grid


def main():
    g = PoleGrid()
    fill_random(g, 0.5, 0.15, 10, (-8, -8))
    fill_random(g, 0.5, 0.15, 12, (8, -10))
    fill_random(g, 0.5, 0.15, 12, (8, 8))
    fill_random(g, 0.5, 0.15, 12, (-10, 8))
    # g.add(NonPole.at_tile((0, 0), True))
    # g.add(NonPole.at_tile((-10, 0), True))
    # g.add(NonPole.at_tile((10, 0), True))

    # g.matplot()

    pole_to_use = small_pole
    possible_poles = g.all_candidate_poles(pole_to_use)
    center = get_center([pole.pos for pole in possible_poles])
    for pole in possible_poles:
        pole.cost = euclid_dist(pole.pos, center) / 1000 + 1

    def cover_cost(pole: CandidatePole) -> float:
        if not pole.covered_entities:
            return pole_to_use.reach
        return pole.cost

    def dist_fun(p1: CandidatePole, p2: CandidatePole) -> float:
        return pole_to_use.reach + manhattan_dist(p1.pos, p2.pos) + cover_cost(p1) + cover_cost(p2)

    center, dists = get_pole_dists(possible_poles, dist_fun)

    def try_with_solver(
            name: str,
            solver: Callable[[list[CandidatePole]], tuple[list[CandidatePole], OptimizeResult]],
    ):
        poles, res = solver(possible_poles)

        g2 = deepcopy(g)

        # ugly, but works for now
        for pole in poles:
            g2.add(Pole(pole.pos, pole_to_use.name))

        g2.matplot(show=False)
        plt.suptitle(name)
        plt.title(f"Num poles: {round(sum(res.x))}")

        adj_list = g2.pole_adj_list()
        the_mst = mst(adj_list)
        plot_adj_list(the_mst)
        plt.show()

        # # plot heatmap, using dists
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.plot(center[0] + 0.5, center[1] + 0.5, 'rX')
        # plt.scatter([pole.pos[0] for pole in possible_poles], [pole.pos[1] for pole in possible_poles],
        #             c=[dists[pole] for pole in possible_poles],
        #             marker='.'
        #             )
        # plt.colorbar()
        # plt.show()

    try_with_solver(
        "Set cover",
        lambda coverages: solve_set_cover(coverages)
    )

    try_with_solver(
        "Approx pole cover",
        lambda coverages: solve_approximate_pole_cover(coverages, dist_fun, milp_options={
            "disp": True,
            "time_limit": 300,
        }))


if __name__ == "__main__":
    main()
