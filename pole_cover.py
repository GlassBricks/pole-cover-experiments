import dataclasses
from heapq import heappush, heappop, heapify
from typing import Callable, Any

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp, OptimizeResult
from scipy.sparse import lil_matrix

from set_cover import get_entity_cov_dict, get_set_cover_constraint

Pos = tuple[int, int]


@dataclasses.dataclass
class CandidatePole:
    pos: Pos
    covered_entities: set[Any]
    pole_neighbors: set[Pos]
    cost: float = 1

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def dijkstras_dist(
        nodes: set[CandidatePole],
        starts: list[CandidatePole],
        dist: Callable[[CandidatePole, CandidatePole], float],
) -> dict[CandidatePole, float]:
    poles_by_pos = {pole.pos: pole for pole in nodes}
    dists = {node: float('inf') for node in nodes}
    vis = set()
    for start in starts:
        dists[start] = 0

    heap = [(0, start.pos) for start in starts if start in nodes]
    heapify(heap)
    while heap:
        d, node_pos = heappop(heap)
        if node_pos in vis:
            continue
        vis.add(node_pos)
        node = poles_by_pos[node_pos]
        for neigh_pos in poles_by_pos[node_pos].pole_neighbors:
            neighbor = poles_by_pos.get(neigh_pos)
            if not neighbor or neighbor in vis:
                continue
            nd = d + dist(node, neighbor)
            if nd < dists[neighbor]:
                dists[neighbor] = nd
                heappush(heap, (nd, neighbor.pos))
    return dists


def manhattan_dist(p1: Pos, p2: Pos) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclid_dist(p1: Pos, p2: Pos) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def euclid_dist_squared(p1: Pos, p2: Pos) -> float:
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def get_center(poles: list[CandidatePole]) -> Pos:
    x = round(sum(pole.pos[0] for pole in poles) / len(poles))
    y = round(sum(pole.pos[1] for pole in poles) / len(poles))
    return x, y


def get_pole_dists(poles, dist_fun):
    center = get_center(poles)
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
        center: Pos,
        dists: dict[CandidatePole, float],
) -> LinearConstraint:
    lt = get_lt(center, dists)

    pole_index = {pole: i for i, pole in enumerate(poles)}
    pos_to_pole = {pole.pos: pole for pole in poles}

    A = lil_matrix((len(poles), len(poles)))
    for i, y in enumerate(poles):
        neighbors = [pos_to_pole[x] for x in y.pole_neighbors if x in pos_to_pole]
        xs = [x for x in neighbors if lt(x, y)]
        if not xs:
            continue
        for x in xs:
            A[i, pole_index[x]] = 1
        A[i, i] = -1

    return LinearConstraint(A, lb=0, ub=np.inf)


def get_lt(center: Pos, dists: dict[CandidatePole, float]) -> Callable[[CandidatePole, CandidatePole], float]:
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
    entity_cov = get_entity_cov_dict(poles)
    cover_constraint = get_set_cover_constraint(poles, entity_cov)
    connectivity_constraint = get_connectivity_constraint(poles, center, dists)

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
