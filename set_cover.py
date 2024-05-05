from collections import defaultdict

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, OptimizeResult
from scipy.sparse import lil_matrix

from pole_grid import CandidatePole


def get_entity_cov_dict(coverages):
    entity_to_cov = defaultdict(list)
    for cov in coverages:
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


def solve_set_cover(
        coverages: list[CandidatePole],
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

    constraint = get_set_cover_constraint(coverages, entity_to_cov)

    bounds = Bounds(0, 1)
    integrality = np.ones_like(c)

    res = milp(
        c=c,
        constraints=[constraint],
        bounds=bounds,
        integrality=integrality,
        options=milp_options
    )

    selected = [
        coverages[i]
        for i, val in enumerate(res.x)
        if val >= 0.9
    ]
    return selected, res
