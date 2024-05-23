from typing import cast

from draftsman.classes.blueprint import Blueprint
from draftsman.classes.entity import Entity
from draftsman.classes.mixins import CircuitConnectableMixin, PowerConnectableMixin
from draftsman.data import entities as Entities
from draftsman.prototypes.electric_pole import ElectricPole
from matplotlib import pyplot as plt

from mst import mst, plot_adj_list
from pole_cover import dist_estimate1, solve_approx_pole_cover
from pole_grid import PoleGrid, Pole, NonPole, aabb_to_bbox, CandidatePole, small_pole

print("importing")
bpStr = open("blueprints/base8.txt").read()
bp = Blueprint(bpStr)

print("processing")

names_to_remove = [p.name for p in [small_pole]]
poles_to_use = [small_pole]

old_count = {}

to_remove = []
g = PoleGrid()
for entity in bp.entities:
    entity = cast(Entity, entity)
    if isinstance(entity, ElectricPole):
        old_count[entity.name] = old_count.get(entity.name, 0) + 1
    if entity.name in names_to_remove:
        to_remove.append(entity)
        continue
    if isinstance(entity, ElectricPole):
        g.add(Pole(
            (entity.position.x, entity.position.y),
            entity.name,
        ), allow_overlap=True)
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

print("old pole counts", old_count)


def remove_circuit_connections(a: CircuitConnectableMixin):
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


old_list = list(bp.entities)
bp.entities.clear()
bp.entity_map.clear()

for el in old_list:
    if el not in to_remove_set:
        bp.entities.append(el, copy=False)
        bp.entity_map.add(el)

for entity in bp.entities:
    if isinstance(entity, CircuitConnectableMixin):
        remove_circuit_connections(entity)
    if isinstance(entity, PowerConnectableMixin):
        entity.neighbours = (
            [neighbor for neighbor in entity.neighbours if neighbor() not in to_remove_set])

# print("visualizing")
# 
# plt.figure(dpi=100, figsize=(10, 10))
# plot_adj_list(mst(g.pole_adj_list()))
# g.matplot()
# 
print("getting candidate poles")

possible_poles = g.all_candidate_poles(*poles_to_use, remove_empty=True, expand=0)

# chests_center = get_center(
#     [entity.pos for entity in g.entities if entity.name == "logistic-chest-storage"]
# )
# for pole in possible_poles:
# cost1 = 1 if pole.pole_type == medium_pole else 5.5
# pole.cost = cost1 # + round(euclid_dist(pole.pos, chests_center)) / 1e6


print("solving")

dist_fun = dist_estimate1()
# poles, res = solve_set_cover(
#     possible_poles,
#     milp_options={
#         "disp": True,
#         "time_limit": 30,
#         "mip_rel_gap": 0.01,
#     }
# )
# print("quick run with set cover (connectivity not guaranteed)")
# print(f"pole count: {counts_by_type(poles)}")

# g2 = deepcopy(g)
# for pole in poles:
#     g2.add(Pole(pole.pos, pole.pole_type))
# plt.figure(dpi=100, figsize=(10, 10))
# plt.suptitle("Set cover")
# plt.title(f"pole count: {counts_by_type(poles)}")
# plot_adj_list(mst(g2.pole_adj_list()))
# g2.matplot()

poles, res = solve_approx_pole_cover(
    possible_poles,
    dist_fun,
    remove_subsets=False,
    remove_equiv=True,
    dist_tolerance=0,
    vis_poles_dist=True,
    milp_options={
        "disp": True,
        "time_limit": 3 * 60,
        "mip_rel_gap": 0.001
    }
)

print("done solving")


def counts_by_type(poles: list[CandidatePole]):
    res = {}
    for pole in poles:
        res[pole.pole_type.name] = res.get(pole.pole_type.name, 0) + 1
    return res


print(f"pole count: {counts_by_type(poles)}")

for pole in poles:
    g.add(pole.orig)
pole_mst = mst(g.pole_adj_list())
num_poles = len(poles)

print("exporting")
bp_pole_by_pos = {}

for pos, entities in g.by_tile.items():
    for entity in entities:
        if isinstance(entity, Pole):
            pole = bp.find_entity_at_position(entity.pos)
            bp_pole_by_pos[entity.pos] = pole

for pole in poles:
    bp_pole = ElectricPole(position={"x": pole.pos[0], "y": pole.pos[1]}, name=pole.pole_type.name, )
    bp.entities.append(bp_pole, copy=False)
    bp_pole_by_pos[pole.orig.pos] = bp_pole

pole_to_index = {pole: i for i, pole in enumerate(poles)}
for pole in poles:
    for neighbor in pole_mst[pole.orig]:
        bp_pole = bp_pole_by_pos[pole.orig.pos]
        bp_neighbor = bp_pole_by_pos[neighbor.pos]
        bp.add_power_connection(bp_pole, bp_neighbor)

# print(bp.to_dict())
str = bp.to_string()
open("output.txt", "w").write(str)

print("final visualizing")
plt.figure(dpi=100, figsize=(10, 10))
plt.suptitle("Pole cover")
plt.title(f"pole count: {counts_by_type(poles)}")
plot_adj_list(pole_mst)
g.matplot()
