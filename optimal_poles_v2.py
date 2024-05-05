from collections import Counter
from copy import deepcopy
from typing import cast

from draftsman.classes.blueprint import Blueprint
from draftsman.classes.entity import Entity
from draftsman.classes.mixins import CircuitConnectableMixin, PowerConnectableMixin
from draftsman.data import entities as Entities
from draftsman.prototypes.electric_pole import ElectricPole
from matplotlib import pyplot as plt

from mst import mst, plot_adj_list
from pole_cover import get_center, euclid_dist, dist_estimate1, solve_approx_pole_cover
from pole_grid import PoleGrid, Pole, NonPole, aabb_to_bbox, medium_pole, substation, CandidatePole
from set_cover import solve_set_cover

bpStr = open("input.txt").read()
print("importing")
bp = Blueprint(bpStr)

names_to_remove = [
    "medium-electric-pole",
]

to_remove = []
g = PoleGrid()
for entity in bp.entities:
    entity = cast(Entity, entity)
    if entity.name in names_to_remove:
        to_remove.append(entity)
        continue
    if isinstance(entity, ElectricPole):
        g.add(Pole(
            (entity.position.x, entity.position.y),
            entity.name,
        ))
    else:
        data = Entities.raw[entity.name]
        powerable = "energy_source" in data and data["energy_source"]["type"] == "electric"
        g.add(NonPole(
            pos=(entity.position.x, entity.position.y),
            bbox=aabb_to_bbox(entity.get_world_bounding_box()),
            powerable=powerable,
            name=entity.name,
        ), allow_overlap=True)

to_remove_set = set(to_remove)


def prune_circuit_connections(
        a: CircuitConnectableMixin,
):
    connections = a.connections

    def remove_pt(pt_list):
        return [
            pt for pt in pt_list if pt["entity_id"]() not in to_remove_set
        ]

    def remove_id(side):
        if "red" in side:
            side["red"] = remove_pt(side["red"])
        if "green" in side:
            side["green"] = remove_pt(side["green"])

    if "1" in connections:
        remove_id(connections["1"])
    if "2" in connections:
        remove_id(connections["2"])


entities = bp.entities
for entity in entities:
    if isinstance(entity, CircuitConnectableMixin):
        prune_circuit_connections(entity)
    if isinstance(entity, PowerConnectableMixin):
        entity.neighbours = (
            [neighbor for neighbor in entity.neighbours if neighbor() not in to_remove_set])

# old_list = list(entities)
# bp.entity_map.clear()
# entities.clear()
# for el in old_list:
#     if el not in to_remove_set:
#         entities.append(el, copy=False)
#     bp.entity_map.add(el)
for entity in to_remove:
    bp.entities.remove(entity)

print("done importing")

plt.figure(dpi=100, figsize=(10, 10))
plot_adj_list(mst(g.pole_adj_list()))
g.matplot()

print("getting candidate poles")

possible_poles = g.all_candidate_poles(medium_pole, substation)

chests_center = get_center(
    [entity.pos for entity in g.entities if entity.name == "logistic-chest-storage"]
)
for pole in possible_poles:
    cost1 = 1 if pole.pole_type == medium_pole else 5.5
    pole.cost = cost1 + round(euclid_dist(pole.pos, chests_center)) / 1e6


def counts_by_type(poles: list[CandidatePole]):
    res = {}
    for pole in poles:
        res[pole.pole_type.name] = res.get(pole.pole_type.name, 0) + 1
    return res


print("solving")

dist_fun = dist_estimate1()
poles, res = solve_set_cover(
    possible_poles,
    milp_options={
        "disp": True,
        "time_limit": 30,
        "mip_rel_gap": 0.01,
    }
)
print("quick run with set cover (connectivity not guaranteed)")
print(f"pole count: {counts_by_type(poles)}")

g2 = deepcopy(g)
for pole in poles:
    g2.add(Pole(pole.pos, pole.pole_type))
plt.figure(dpi=100, figsize=(10, 10))
plt.suptitle("Set cover")
plt.title(f"Num poles: {len(poles)}")
plot_adj_list(mst(g2.pole_adj_list()))
g2.matplot()

poles, res = solve_approx_pole_cover(
    possible_poles,
    dist_fun,
    milp_options={
        "disp": True,
        "time_limit": 180,
        "mip_rel_gap": 0.005,
    }
)

print("done solving")
print(f"pole count: {counts_by_type(poles)}")

for pole in poles:
    g.add(Pole(pole.pos, pole.pole_type))

num_poles = len(poles)

plt.figure(dpi=100, figsize=(10, 10))
plt.suptitle("Pole cover")
plt.title(f"pole count: {counts_by_type(poles)}")
plot_adj_list(mst(g.pole_adj_list()))
g.matplot()

print("exporting")
for pole in poles:
    bp.entities.append(
        ElectricPole(
            position={"x": pole.pos[0], "y": pole.pos[1]},
            name=pole.pole_type.name,
        )
    )

str = bp.to_string()
open("output2.txt", "w").write(str)
