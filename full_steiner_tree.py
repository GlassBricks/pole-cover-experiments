# directed flow formulation for steiner tree
# modified so weights are on non-terminal nodes, not edges
# and any pre-connected pole can be considered root
from typing import Iterable

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import lil_matrix

from pole_grid import CandidatePole, PoleGrid, Entity
from set_cover import get_entity_cov_dict


class Var:
    name: str
    integral: bool
    lb: float
    ub: float

    def __init__(self, name: str,
                 integral: bool = True,
                 lb: float = 0, ub: float = 1):
        self.name = name
        self.integral = integral
        self.lb = lb
        self.ub = ub


class ConstraintEquation:
    weights: dict[Var, float]
    min: float
    max: float

    def __init__(self, weights: dict[Var, float], min: float, max: float):
        self.weights = weights
        self.min = min
        self.max = max


class AbstractIlp:
    all_vars: list[Var]
    var_to_index: dict[Var, int]
    constraints: list[ConstraintEquation]
    cost_eq: dict[Var, float]

    def __init__(self):
        self.all_vars = []
        self.var_to_index = {}
        self.constraints = []
        self.cost_eq = None

    def var(self, name: str, integral: bool = True, lb: float = 0, ub: float = 1):
        var = Var(name, integral, lb, ub)
        self.all_vars.append(var)
        self.var_to_index[var] = len(self.all_vars) - 1
        return var

    def constraint(self, weights: dict[Var, float], lb: float, ub: float):
        self.constraints.append(ConstraintEquation(weights, lb, ub))

    def cost(self, costs: dict[Var, float]):
        self.cost_eq = costs

    def to_milp_params(self):
        if not self.cost:
            raise ValueError("Cost equation not set")

        A = lil_matrix((len(self.constraints), len(self.all_vars)))
        lb = np.zeros(len(self.constraints))
        ub = np.zeros(len(self.constraints))

        for i, constraint in enumerate(self.constraints):
            for var, weight in constraint.weights.items():
                A[i, self.var_to_index[var]] = weight
            lb[i] = constraint.min
            ub[i] = constraint.max

        constraints = [LinearConstraint(A, lb=lb, ub=ub)]

        c = np.array([self.cost_eq.get(var, 0) for var in self.all_vars])
        bounds = Bounds(np.array([var.lb for var in self.all_vars]), np.array([var.ub for var in self.all_vars]))
        integrality = np.array([var.integral for var in self.all_vars])

        return {
            "c": c,
            "constraints": constraints,
            "bounds": bounds,
            "integrality": integrality,
        }

    def solve(self, **kwargs):
        return milp(**self.to_milp_params(), **kwargs)


def steiner_flow(
        poles: list[CandidatePole],
        root_poles: Iterable[CandidatePole],
):
    ilp = AbstractIlp()

    pole_to_index = {pole: i for i, pole in enumerate(poles)}
    pole_orig_to_index = {pole.orig: i for i, pole in enumerate(poles)}
    pole_vars = [ilp.var(f"x_{i}") for i in range(len(poles))]
    root_pole_i = {pole_to_index[pole] for pole in root_poles}

    entity_cov_map = get_entity_cov_dict(poles)
    all_entities = sorted(entity_cov_map.keys(), key=lambda x: x.pos)
    entity_to_index = {entity: i for i, entity in enumerate(all_entities)}

    g = PoleGrid()
    for pole in poles:
        g.add(pole.orig)
    pole_graph = g.pole_adj_list()

    pole_edges = [
        (pole_to_index[pole], pole_orig_to_index[neigh_orig])
        for pole in poles
        for neigh_orig in pole_graph[pole.orig]
    ]

    edge_idx_by_pole_idx = {}
    in_edges = {i: [] for i in range(len(poles))}
    out_edges = {i: [] for i in range(len(poles))}
    for k, (i, j) in enumerate(pole_edges):
        edge_idx_by_pole_idx[i, j] = k
        in_edges[j].append((k, i))
        out_edges[i].append((k, j))

    def add_vars_for_entity(target: Entity):
        e_index = entity_to_index[target]
        hitting_poles = [pole_to_index[pole] for pole in entity_cov_map[target]]
        cover_vars = {
            i: ilp.var(f"c_{e_index}_{i}")
            for i in hitting_poles
        }
        hitting_poles_set = set(hitting_poles)

        ilp.constraint({var: 1 for _, var in cover_vars.items()}, 1, np.inf)

        flow_vars = [
            ilp.var(f"f_{i}_{j}")
            for i, j in pole_edges
        ]

        # flow only active if pole is in cover
        for k, (i, j) in enumerate(pole_edges):
            # f_i_j <= x_i, x_j
            # x_i - f_i_j <= 0
            ilp.constraint({pole_vars[i]: 1, flow_vars[k]: -1}, 0, np.inf)
            ilp.constraint({pole_vars[j]: 1, flow_vars[k]: -1}, 0, np.inf)

        # flow only goes in one direction
        for k, (i, j) in enumerate(pole_edges):
            if i > j:
                continue
            k_rev = edge_idx_by_pole_idx[j, i]
            # f_i_j + f_j_i <= 1
            ilp.constraint({flow_vars[k]: 1, flow_vars[k_rev]: 1}, 0, 1)

        # flow into pole = flow out of pole; except for root poles
        # cover_vars count as 1 outflow
        for i in range(len(poles)):
            if i in root_pole_i:
                continue
            flow = {flow_vars[k]: 1 for k, j in in_edges[i]}
            flow.update({flow_vars[k]: -1 for k, j in out_edges[i]})
            if i in hitting_poles_set:
                flow.update({cover_vars[i]: -1})
            ilp.constraint(flow, 0, 0)

        # root_poles should have no inflow
        for i in range(len(poles)):
            if i in root_pole_i:
                ilp.constraint({flow_vars[k]: 1 for k, j in in_edges[i]}, 0, 0)

    for entity in all_entities:
        add_vars_for_entity(entity)

    ilp.cost(
        {pole_vars[i]: pole.cost for i, pole in enumerate(poles)}
    )

    return ilp, pole_vars


def solve_pole_cover_steiner_flow(
        poles: list[CandidatePole],
        root_poles: Iterable[CandidatePole],
        **kwargs,
):
    ilp, pole_vars = steiner_flow(poles, root_poles)
    pole_idxes = [
        ilp.var_to_index[var]
        for var in pole_vars
    ]
    res = ilp.solve(**kwargs)

    poles = [
        poles[i]
        for i in pole_idxes
        if res.x[i] > 0.5
    ]
    return poles, res
