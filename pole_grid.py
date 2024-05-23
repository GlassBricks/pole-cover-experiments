from abc import abstractmethod, ABC
from dataclasses import dataclass
from math import ceil, floor
from typing import Generator, Iterable, Tuple, Any, Union

from draftsman.utils import AABB
from matplotlib import pyplot as plt

Pos = tuple[int, int]
FPos = tuple[float, float]
BBox = Tuple[FPos, FPos]


@dataclass(frozen=True)
class PoleType:
    name: str
    coverage: int
    reach: float
    size: int = 1


pole_types = {}

from draftsman.data import entities


def get_pole_type(name: str) -> PoleType:
    if name in pole_types:
        return pole_types[name]
    data = entities.raw[name]
    reach = data.get("maximum_wire_distance")
    coverage = data.get("supply_area_distance")
    ((lx, ly), (hx, hy)) = data.get("collision_box")
    height = data.get("tile_width") or ceil(hy - ly)
    width = data.get("tile_height") or ceil(hx - lx)
    if width != height:
        raise ValueError("Pole bounding box must be square")
    size = width
    res = PoleType(name, coverage, reach, size)
    pole_types[name] = res
    return res


small_pole = get_pole_type("small-electric-pole")
medium_pole = get_pole_type("medium-electric-pole")
big_pole = get_pole_type("big-electric-pole")
substation = get_pole_type("substation")


def iterate_bbox_tiles(bbox: BBox) -> Generator[Pos, None, None]:
    (lx, ly), (hx, hy) = bbox
    for x in range(floor(lx), ceil(hx)):
        for y in range(floor(ly), ceil(hy)):
            yield x, y


class Entity(ABC):

    def __init__(self, pos: FPos, name: str):
        self.pos = pos
        self.name = name

    @property
    @abstractmethod
    def bbox(self) -> BBox:
        pass

    def occupied_tiles(self) -> Iterable[Pos]:
        return iterate_bbox_tiles(self.bbox)


class Pole(Entity):

    def __init__(self, pos: FPos, pole_type: Union[str, PoleType]):
        if isinstance(pole_type, PoleType):
            name = pole_type.name
        else:
            name = pole_type
        super().__init__(pos, name)
        self.check_alignment()

    def check_alignment(self):
        if self.pole_type.size % 2 == 1:
            # must be aligned to 0.5
            if self.pos[0] % 1 != 0.5 or self.pos[1] % 1 != 0.5:
                raise ValueError("Pole must be aligned to grid")
        else:
            if self.pos[0] % 1 != 0 or self.pos[1] % 1 != 0:
                raise ValueError("Pole must be aligned to grid")

    @property
    def pole_type(self) -> PoleType:
        return get_pole_type(self.name)

    @property
    def bbox(self) -> BBox:
        px, py = self.pos
        size = self.pole_type.size
        return (px - size / 2, py - size / 2), (px + size / 2, py + size / 2)

    def powered_tiles(self) -> Generator[Pos, None, None]:
        coverage = self.pole_type.coverage
        px, py = self.pos
        for x in range(floor(px - coverage), ceil(px + coverage)):
            for y in range(floor(py - coverage), ceil(py + coverage)):
                yield x, y

    def reachable_tiles(self) -> Generator[Pos, None, None]:
        reach = self.pole_type.reach
        px, py = self.pos
        for x in range(floor(px - reach), ceil(px + reach)):
            for y in range(floor(py - reach), ceil(py + reach)):
                xx, yy = x + 0.5, y + 0.5
                if (px - xx) ** 2 + (py - yy) ** 2 <= reach ** 2:
                    yield x, y

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, pos={self.pos})"


def aabb_to_bbox(aabb: AABB) -> BBox:
    (lx, ly), (hx, hy) = aabb.top_left, aabb.bot_right
    return (lx, ly), (hx, hy)


class NonPole(Entity):
    pos: FPos
    bbox: BBox

    def __init__(self,
                 pos: FPos,
                 bbox: BBox,
                 powerable: bool,
                 name: str = "non-pole"):
        super().__init__(pos, name)
        self._bbox = bbox
        self.powerable = powerable
        self.name = name

    @property
    def bbox(self):
        return self._bbox

    @classmethod
    def at_tile(cls, pos: Pos, powerable: bool, name: str = "non-pole"):
        return cls(
            (pos[0] + 0.5, pos[1] + 0.5),
            (pos, (pos[0] + 1, pos[1] + 1)), powerable, name)

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, pos={self.pos}, powerable={self.powerable})"
    
    def __repr__(self):
        return str(self)


class CandidatePole:
    orig: Pole
    powered_entities: set[Entity]
    connects_to_existing: bool
    pole_neighbors: set["CandidatePole"]
    cost: float

    def __init__(
            self,
            orig: Pole,
            powered_entities: set[Any],
            connects_to_existing: bool,
            pole_neighbors=None,
            cost: float = 1,
    ):
        if pole_neighbors is None:
            pole_neighbors = set()
        self.orig = orig
        self.powered_entities = powered_entities
        self.pole_neighbors = pole_neighbors
        self.connects_to_existing = connects_to_existing
        self.cost = cost

    @property
    def pos(self):
        return self.orig.pos

    @property
    def pole_type(self):
        return self.orig.pole_type

    def __repr__(self):
        return (f"{self.__class__.__name__}<pos={self.pos}, covered={self.powered_entities}), cost={self.cost}, "
                f"neighbors(pos)={[n.pos for n in self.pole_neighbors]}>")


class PoleGrid:
    entities: set[Entity]
    by_tile: dict[Pos, set[Entity]]

    def __init__(self):
        self.entities = set()
        self.by_tile = {}

    def add(self, entity: Entity, allow_overlap: bool = False):
        self.entities.add(entity)
        for pos in entity.occupied_tiles():
            if not allow_overlap and self.by_tile.get(pos):
                raise ValueError("Entity overlaps with existing entity: " + str(entity), str(self.by_tile[pos]))
            self.by_tile.setdefault(pos, set()).add(entity)

    def remove(self, entity: Entity):
        self.entities.remove(entity)
        for pos in entity.occupied_tiles():
            if not self.by_tile.get(pos):
                continue
            self.by_tile[pos].remove(entity)
            if not self.by_tile[pos]:
                del self.by_tile[pos]

    def __contains__(self, item):
        return item in self.entities

    def get_bounds(self, expand: float = 0) -> BBox:
        bboxes = [entity.bbox for entity in self.entities]
        min_x = min(b[0][0] for b in bboxes)
        max_x = max(b[1][0] for b in bboxes)
        min_y = min(b[0][1] for b in bboxes)
        max_y = max(b[1][1] for b in bboxes)
        return (floor(min_x - expand), floor(min_y - expand)), (ceil(max_x + expand), ceil(max_y + expand))

    def can_place_pole(self, pole: Pole) -> bool:
        # for pos in pole.occupied_tiles():
        #     if self.by_tile[pos]:
        #         return False
        # return True
        return not any(self.by_tile.get(pos) for pos in pole.occupied_tiles())

    def possible_poles(self, pole_type: PoleType, bbox: BBox) -> Generator[Pole, None, None]:
        (lx, ly), (hx, hy) = bbox
        lx, ly, hx, hy = floor(lx), floor(ly), ceil(hx), ceil(hy)
        size = pole_type.size
        for x in range(lx, hx - size + 1):
            for y in range(ly, hy - size + 1):
                xx = x + size / 2
                yy = y + size / 2
                pole = Pole((xx, yy), pole_type.name)
                if self.can_place_pole(pole):
                    yield pole

    def powered_entities(self, pole: Pole) -> set[Entity]:
        return {
            entity
            for pos in pole.powered_tiles()
            if pos in self.by_tile
            for entity in self.by_tile[pos]
            if isinstance(entity, NonPole) and entity.powerable
        }

    def pole_neighbors(self, pole: Pole) -> Generator[Pole, None, None]:
        """
        Returns all poles in this graph that are within reach of the given pole.
        Given pole may or may not be in the graph.
        """
        pos = pole.pos
        for tile_pos in pole.reachable_tiles():
            if tile_pos not in self.by_tile:
                continue
            for entity in self.by_tile[tile_pos]:
                if (entity != pole
                        and isinstance(entity, Pole)
                        and (entity.pos[0] - pos[0]) ** 2 + (entity.pos[1] - pos[1]) ** 2
                        <= min(pole.pole_type.reach, entity.pole_type.reach) ** 2):
                    yield entity

    def pole_adj_list(self) -> dict[Pole, set[Pole]]:
        return {
            pole: set(self.pole_neighbors(pole))
            for pole in self.entities
            if isinstance(pole, Pole)
        }

    # does not include already powered stuff
    def all_candidate_poles(
            self,
            *pole_types: PoleType,
            expand: float = 0,
            remove_empty = False,
    ) -> list[CandidatePole]:

        already_powered = {
            entity
            for pole in self.entities
            if isinstance(pole, Pole)
            for pos in pole.powered_tiles()
            if pos in self.by_tile
            for entity in self.by_tile[pos]
            if isinstance(entity, NonPole) and entity.powerable
        }

        bounds = self.get_bounds(expand=expand)
        poles = [
            pole for pole_type in pole_types
            for pole in self.possible_poles(pole_type, bounds)
        ]
        cand_poles = {
            pole: CandidatePole(
                orig=pole,
                powered_entities=self.powered_entities(pole) - already_powered,
                connects_to_existing=any(self.pole_neighbors(pole)),
            )
            for pole in poles
        }
        if remove_empty:
            cand_poles = {
                pole: candidate
                for pole, candidate in cand_poles.items()
                if candidate.powered_entities
            }
        pole_graph = PoleGrid()
        for pole, _ in cand_poles.items():
            pole_graph.add(pole, allow_overlap=True)
        for pole, candidate in cand_poles.items():
            candidate.pole_neighbors = [
                cand_poles[neighbor]
                for neighbor in pole_graph.pole_neighbors(pole)
                if neighbor in cand_poles
            ]
        return [cand_poles[pole] for pole in poles if pole in cand_poles]

    def matplot(self, show: bool = True):
        for entity in self.entities:
            x, y = entity.pos
            if isinstance(entity, Pole):
                plt.plot(x, y, 'ro')
            elif isinstance(entity, NonPole):
                if entity.powerable:
                    plt.plot(x, y, 'gx')
                else:
                    plt.plot(x, y, 'b.')
        (min_x, min_y), (max_x, max_y) = self.get_bounds(expand=1)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()
