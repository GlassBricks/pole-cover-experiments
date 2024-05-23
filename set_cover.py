from collections import defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult
from scipy.sparse import lil_matrix

from pole_grid import CandidatePole, PoleGrid, Entity


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
