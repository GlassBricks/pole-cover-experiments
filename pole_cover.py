from dataclasses import dataclass
from heapq import heappush, heappop, heapify
from typing import Callable, Any

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp, OptimizeResult
from scipy.sparse import lil_matrix

from pole_graph import FPos, CandidatePole
from set_cover import get_entity_cov_dict, get_set_cover_constraint


@dataclass()
class QEntry:
    d: float
    pole: CandidatePole

    def __lt__(self, other):
        return self.d < other.d

    def __iter__(self):
        return iter((self.d, self.pole))


def dijkstras_dist(
        nodes: set[CandidatePole],
        starts: list[CandidatePole],
        dist: Callable[[CandidatePole, CandidatePole], float],
) -> dict[CandidatePole, float]:
    dists = {node: float('inf') for node in nodes}
    vis = set()
    for start in starts:
        dists[start] = 0

    heap = [QEntry(0, start) for start in starts if start in nodes]
    heapify(heap)
    while heap:
        d, node = heappop(heap)
        if node in vis:
            continue
        vis.add(node)
        for neighbor in node.pole_neighbors:
            if not neighbor or neighbor in vis:
                continue
            nd = d + dist(node, neighbor)
            if nd < dists[neighbor]:
                dists[neighbor] = nd
                heappush(heap, QEntry(nd, neighbor))
    return dists


def manhattan_dist(p1: FPos, p2: FPos) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclid_dist(p1: FPos, p2: FPos) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def euclid_dist_squared(p1: FPos, p2: FPos) -> float:
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def get_center(poles: list[FPos]) -> FPos:
    x = round(sum(pole[0] for pole in poles) / len(poles))
    y = round(sum(pole[1] for pole in poles) / len(poles))
    return x, y


def get_pole_dists(
        poles: list[CandidatePole],
        dist_fun: Callable[[CandidatePole, CandidatePole], float]
) -> tuple[FPos, dict[CandidatePole, float]]:
    center = get_center([pole.pos for pole in poles])
    centermost_pole = min([pole for pole in poles if pole.covered_entities],
                          key=lambda x: euclid_dist_squared(x.pos, center))
    some_entity = next(iter(centermost_pole.covered_entities))
    starts = [pole for pole in poles if some_entity in pole.covered_entities]
    dists = dijkstras_dist(set(poles), starts, dist_fun)
    if len(dists) != len(poles):
        raise ValueError("Graph is not connected")
    return center, dists


def get_connectivity_constraint(
        poles: list[CandidatePole],
        center: FPos,
        dists: dict[CandidatePole, float],
) -> LinearConstraint:
    lt = get_lt(center, dists)

    pole_index = {pole: i for i, pole in enumerate(poles)}

    A = lil_matrix((len(poles), len(poles)))
    for i, y in enumerate(poles):
        xs = [x for x in y.pole_neighbors if lt(x, y)]
        if not xs:
            continue
        for x in xs:
            A[i, pole_index[x]] = 1
        A[i, i] = -1

    return LinearConstraint(A, lb=0, ub=np.inf)


def get_lt(center: FPos, dists: dict[CandidatePole, float]) -> Callable[[CandidatePole, CandidatePole], float]:
    def lt(a: CandidatePole, b: CandidatePole) -> float:
        if dists[a] != dists[b]:
            return dists[a] < dists[b]
        return euclid_dist_squared(a.pos, center) < euclid_dist_squared(b.pos, center)

    return lt


def solve_approximate_pole_cover(
        poles: list[CandidatePole],
        dist_fun: Callable[[CandidatePole, CandidatePole], float],
        milp_options: dict[str, Any] = None
) -> tuple[list[CandidatePole], OptimizeResult]:
    center, dists = get_pole_dists(poles, dist_fun)
    connectivity_constraint = get_connectivity_constraint(poles, center, dists)
    entity_cov = get_entity_cov_dict(poles)
    cover_constraint = get_set_cover_constraint(poles, entity_cov)

    c = np.array([pole.cost for pole in poles])
    bounds = Bounds(0, 1)
    integrality = np.ones_like(c)

    # print(connectivity_constraint.A *)

    res: OptimizeResult = milp(
        c=c,
        constraints=[cover_constraint, connectivity_constraint],
        bounds=bounds,
        integrality=integrality,
        options=milp_options
    )
    if res.status == 2:
        print("Infeasible")

    selected = [
        poles[i]
        for i, val in enumerate(res.x)
        if val >= 0.9
    ]

    return selected, res
