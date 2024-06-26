from collections import defaultdict
from dataclasses import dataclass
from heapq import heappush, heappop, heapify
from math import sqrt
from typing import Callable, Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import LinearConstraint, milp, OptimizeResult, Bounds
from scipy.sparse import lil_matrix

from pole_grid import FPos, CandidatePole, Entity, PoleGrid


# Pole cover, in theory, is a weird version of steiner tree.
# However, with a large number of terminals, even with approximation algs its intractable to solve exactly
# This is a heuristic approach; still NP complete but good using an ILP solver in most cases.
# First, build a DAG based on the graph distance to existing poles if given, or else some central entity, using some
# graph distance metric. Then solve set cover wih an ILP with an added "connectivity" constraint.
# If a pole is used, it must connect to some pole closer to the center (graph-wise, not geometrically).
# Ideally, in most cases, forcing the connection to connect to the graph center is good enough.
# This also has the side effect of discouraging "fragile" connections (really long paths when shorter paths exist)


def get_entity_cov_dict(poles: list[CandidatePole]) -> dict[Entity, list[CandidatePole]]:
    entity_to_cov = defaultdict(list)
    for cov in poles:
        for entity in cov.powered_entities:
            entity_to_cov[entity].append(cov)
    return entity_to_cov


def get_set_cover_constraint(sets, entity_to_set):
    num_entities = len(entity_to_set)
    num_sets = len(sets)
    cov_id = {cov: i for i, cov in enumerate(sets)}
    A = lil_matrix((num_entities, num_sets))
    for i, entity in enumerate(entity_to_set):
        for coverage in entity_to_set[entity]:
            if coverage in cov_id:
                A[i, cov_id[coverage]] = 1
    return LinearConstraint(A, lb=1, ub=np.inf)


def get_non_intersecting_constraint(poles: list[CandidatePole]):
    pole_to_candidate = {p.orig: p for p in poles}
    pole_to_index = {p: i for i, p in enumerate(poles)}
    grid = PoleGrid()
    for pole in poles:
        grid.add(pole.orig, allow_overlap=True)
    overlap_sets = []
    for pos, pos_poles in grid.by_tile.items():
        if len(poles) <= 1:
            continue
        overlap_sets.append([pole_to_candidate[p] for p in pos_poles])
    A = lil_matrix((len(overlap_sets), len(poles)))
    for i, poles in enumerate(overlap_sets):
        for pole in poles:
            A[i, pole_to_index[pole]] = 1
    return LinearConstraint(A, lb=0, ub=1)


def remove_unreferenced_poles(
        poles: set[CandidatePole],
):
    for pole in poles:
        pole.pole_neighbors = [p for p in pole.pole_neighbors if p in poles]


def remove_subset_poles(
        candidate_poles: list[CandidatePole],
        equiv_only: bool = False,
        keep_empty: bool = False,
) -> list[CandidatePole]:
    res = []
    entity_to_cov = get_entity_cov_dict(candidate_poles)
    for pole in sorted(candidate_poles, key=lambda x: len(x.powered_entities), reverse=True):
        entities = pole.powered_entities
        if not entities:
            if keep_empty:
                res.append(pole)
            continue
        e1 = next(iter(entities))
        if any(
                other != pole and
                other.pole_type == pole.pole_type and
                ((not equiv_only and entities < other.powered_entities)
                 or entities == other.powered_entities and (
                         len(pole.pole_neighbors) < len(other.pole_neighbors)
                         # pole.cost >= other.cost
                 )
                )
                for other in entity_to_cov[e1]
        ):
            continue
        res.append(pole)
    remove_unreferenced_poles(set(res))
    return res


@dataclass()
class QEntry:
    d: float
    pole: CandidatePole

    def __lt__(self, other):
        return self.d < other.d

    def __iter__(self):
        return iter((self.d, self.pole))


def dijkstras_dist(
        nodes: list[CandidatePole],
        starts: list[CandidatePole],
        dist: Callable[[CandidatePole, CandidatePole], float],
) -> dict[CandidatePole, float]:
    dists = {node: float('inf') for node in nodes}
    vis = set()
    for start in starts:
        dists[start] = 0

    heap = [QEntry(0, start) for start in starts]
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


def dist_estimate1() -> Callable[[CandidatePole, CandidatePole], float]:
    def cover_cost(pole: CandidatePole) -> float:
        if not pole.powered_entities:
            return sqrt(pole.pole_type.reach)
        return pole.cost

    def dist_fun(p1: CandidatePole, p2: CandidatePole) -> float:
        return (min(p1.pole_type.reach, p2.pole_type.reach) / 10 + euclid_dist(p1.pos, p2.pos)
                + cover_cost(p1) + cover_cost(p2))

    return dist_fun


def get_root_poles(
        poles: list[CandidatePole],
) -> list[CandidatePole]:
    return [max(poles, key=lambda x: x.pos)]
    # max(poles, key=lambda x: x.pos).connects_to_existing = True
    # starts = [pole for pole in poles if pole.connects_to_existing]
    # 
    # if starts:
    #     return starts

    center = get_center([pole.pos for pole in poles])
    centermost_pole = min([pole for pole in poles if pole.powered_entities],
                          key=lambda x: euclid_dist_squared(x.pos, center))
    some_entity = next(iter(centermost_pole.powered_entities))
    return [pole for pole in poles if some_entity in pole.powered_entities]


def get_pole_dists(
        poles: list[CandidatePole],
        dist_fun: Callable[[CandidatePole, CandidatePole], float]
) -> dict[CandidatePole, float]:
    starts = get_root_poles(poles)
    dists = dijkstras_dist(poles, starts, dist_fun)
    if len(dists) != len(poles):
        raise ValueError("Graph is not connected")
    return dists


def get_connectivity_constraint(poles: list[CandidatePole], dists: dict[CandidatePole, float],
                                dist_tolerance: float) -> LinearConstraint:
    def can_reach(a: CandidatePole, b: CandidatePole) -> float:
        return dists[a] < dists[b] + dist_tolerance

    pole_index = {pole: i for i, pole in enumerate(poles)}

    A = lil_matrix((len(poles), len(poles)))
    for i, y in enumerate(poles):
        xs = [x for x in y.pole_neighbors if can_reach(x, y)]
        if not xs:
            continue
        for x in xs:
            A[i, pole_index[x]] = 1
        A[i, i] = -1

    return LinearConstraint(A, lb=0, ub=np.inf)


def do_vis_presolve(
        poles: list[CandidatePole],
        dists: dict[CandidatePole, float],
):
    # print dists as heatmap
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter([pole.pos[0] for pole in poles], [pole.pos[1] for pole in poles],
                c=[dists[pole] for pole in poles],
                marker='.'
                )
    plt.colorbar()
    plt.show()


def solve_approx_pole_cover(
        poles: list[CandidatePole],
        dist_fun: Callable[[CandidatePole, CandidatePole], float],
        *,
        dist_tolerance=0,
        remove_subsets: bool = False,
        remove_equiv: bool = False,
        vis_poles_dist: bool = False,
        milp_options: dict[str, Any] = None,
) -> tuple[list[CandidatePole], OptimizeResult]:
    if remove_subsets or remove_equiv:
        poles = remove_subset_poles(poles, equiv_only=remove_equiv and not remove_subsets)

    dists = get_pole_dists(poles, dist_fun)
    connectivity_constraint = get_connectivity_constraint(poles, dists, dist_tolerance)
    entity_cov = get_entity_cov_dict(poles)
    cover_constraint = get_set_cover_constraint(poles, entity_cov)
    non_intersecting_constraint = get_non_intersecting_constraint(poles)

    if vis_poles_dist:
        do_vis_presolve(poles, dists)

    # print("connectivity:", connectivity_constraint.A.shape)
    # print("cover:", cover_constraint.A.shape)
    c = np.array([pole.cost for pole in poles])

    res: OptimizeResult = milp(
        c=c,
        constraints=[cover_constraint, connectivity_constraint, non_intersecting_constraint],
        bounds=Bounds(0, 1),
        integrality=1,
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


def solve_set_cover(
        coverages: list[CandidatePole],
        *,
        remove_subsets: bool = False,
        milp_options: dict[str, any] = None
) -> tuple[list[CandidatePole], OptimizeResult]:
    entity_to_cov = get_entity_cov_dict(coverages)

    # remove subsets; only look at poles that share at least one entity
    # with another pole

    if remove_subsets:
        new_coverages = []
        for coverage in sorted(coverages, key=lambda x: len(x.powered_entities), reverse=True):
            entities = coverage.powered_entities
            if not entities:
                break
            e1 = next(iter(entities))
            if any(entities < p.powered_entities for p in (entity_to_cov[e1])):
                continue
            new_coverages.append(coverage)
        coverages = new_coverages

    c = np.array([cov.cost for cov in coverages])

    constraints = [
        get_set_cover_constraint(coverages, entity_to_cov),
        get_non_intersecting_constraint(coverages)
    ]

    res = milp(
        c=c,
        constraints=constraints,
        bounds=Bounds(0, 1),
        integrality=1,
        options=milp_options
    )

    selected = [
        coverages[i]
        for i, val in enumerate(res.x)
        if val >= 0.9
    ]
    return selected, res
