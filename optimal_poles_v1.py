from math import floor, ceil

from draftsman.blueprintable import Blueprint
from draftsman.classes.entitylike import EntityLike
from draftsman.classes.mixins import CircuitConnectableMixin
from draftsman.classes.vector import Vector
from draftsman.prototypes.electric_pole import ElectricPole
from draftsman.utils import AABB

from pole_cover import get_center, euclid_dist, solve_approx_pole_cover, manhattan_dist
from set_cover import solve_set_cover
from pole_grid import CandidatePole

bpStr = open("input.txt").read()
print("importing")
bp = Blueprint(bpStr)

pole_reaches = {
    "small-electric-pole": 5,
    "medium-electric-pole": 7,
    "big-electric-pole": 4,
    "substation": 18,
}

print("preprocessing")

pole_to_use = "medium-electric-pole"
assert pole_to_use in pole_reaches

entities = [
    entity for entity in bp.entities
    if entity.name != pole_to_use
]
i = 0
for entity in entities:
    entity.id = f"{entity.name}-{entity.position}-{i}"
    i += 1
    if (isinstance(entity, CircuitConnectableMixin)):
        entity.connections = {}
    if (isinstance(entity, ElectricPole)):
        entity.neighbours = []

already_connected = set()
for entity in entities:
    if not isinstance(entity, ElectricPole):
        continue
    pole_reach = pole_reaches[entity.name]
    coverage = AABB(
        entity.position.x - pole_reach / 2, entity.position.y - pole_reach / 2,
        entity.position.x + pole_reach / 2, entity.position.y + pole_reach / 2)
    covered_entities: list[EntityLike] = bp.entity_map.get_in_area(coverage)
    for covered_entity in covered_entities:
        already_connected.add(covered_entity.id)

bp.entities = entities
bbox: AABB = bp.area

powerable_types = {
    "inserter",
    "assembling-machine",
    "decider-combinator",
    "arithmetic-combinator",
    "roboport",
    "radar",
    "rocket-silo",
    "programmable-speaker",
    "pump",
    "beacon"
}
powerable_names = {
    "electric-furnace",
    "electric-mining-drill",
}


def maybe_add_pole(position: Vector):
    pole_reach = pole_reaches[pole_to_use]
    if bp.entity_map.get_on_point(pole_position, limit=1):
        return
    coverage = AABB(
        pole_position.x - pole_reach / 2, pole_position.y - pole_reach / 2,
        pole_position.x + pole_reach / 2, pole_position.y + pole_reach / 2)
    covered_entities: list[EntityLike] = bp.entity_map.get_in_area(coverage)
    this_entities = {
        entity.id for entity in covered_entities
        if (entity.type in powerable_types or entity.name in powerable_names)
           and entity.id not in already_connected
    }
    # if not this_entities:
    #     return
    return CandidatePole(position, this_entities, set(), False)


def remove_subset_poles(candidate_poles: list[Coverage]) -> list[Coverage]:
    # remove poles that are a strict subset of another pole
    res = []
    for pole in sorted(candidate_poles, key=lambda x: len(x.powered_entities), reverse=True):
        if any(pole.powered_entities.issubset(other.powered_entities) for other in res):
            continue
        res.append(pole)
    return res


print("getting candidate poles")
candidate_poles: list[Coverage] = []
for x in range(floor(bbox.world_top_left[0]), ceil(bbox.world_bot_right[0])):
    for y in range(floor(bbox.world_top_left[1]), ceil(bbox.world_bot_right[1])):
        pole_position = Vector(x + 0.5, y + 0.5)
        if pole := maybe_add_pole(pole_position):
            candidate_poles.append(pole)

pos_to_pole = {pole.pos: pole for pole in candidate_poles}

center = get_center(candidate_poles)
for pole in candidate_poles:
    pole.cost = 50 + round(euclid_dist(pole.pos, center))
    # get all poles within reach

print("Solving")

poles, res = solve_set_cover(candidate_poles, {
    'disp': True,
})

# poles, res = solve_approximate_pole_cover(
#     candidate_poles,
#     dist_fun=lambda x, y: 10 + manhattan_dist(x.pos, y.pos),
#     milp_options={
#         'disp': True,
#     })
# 
# create blueprint
for pole in poles:
    bp.entities.append(ElectricPole(
        name=pole_to_use,
        position=pole.pos,
    ))

import pyperclip

str = bp.to_string()
pyperclip.copy(str)
open("output2.txt", "w").write(str)
