from math import ceil, floor
from typing import Generator

from matplotlib import pyplot as plt

from pole_cover import CandidatePole
from set_cover import Pos

AdjList = dict[Pos, set[Pos]]
Blocker = -1
Pole = 0
NodeType = int


class PoleGraph:
    nodes: dict[Pos, int]
    pole_reach: float  # euclidean distance
    pole_coverage: int  # manhattan distance

    def __init__(self, reach: float, coverage: int):
        self.nodes = {}
        self.pole_reach = reach
        self.pole_coverage = coverage

    def pole_neighbors(self, pole: Pos) -> Generator[Pos, None, None]:
        for neighbor in self.possible_pole_neighbors(pole):
            if self.nodes.get(neighbor) == Pole:
                yield neighbor

    def possible_pole_neighbors(self,
                                pole: Pos,
                                bounds: tuple[int, int, int, int] = None
                                ) -> Generator[Pos, None, None]:
        lx, hx, ly, hy = bounds or (-float('inf'), float('inf'), -float('inf'), float('inf'))
        px, py = pole
        for x in range(max(ceil(px - self.pole_reach), lx), min(floor(px + self.pole_reach), hx) + 1):
            for y in range(max(ceil(py - self.pole_reach), ly), min(floor(py + self.pole_reach), hy) + 1):
                if (
                        (x != px or y != py) and
                        (px - x) ** 2 + (py - y) ** 2 <= self.pole_reach ** 2 and
                        self.nodes.get((x, y)) in [None, Pole]
                ):
                    yield x, y

    def pole_adj_list(self) -> AdjList:
        adj_list = {}
        for node in self.nodes:
            if self.nodes[node] == Pole:
                adj_list[node] = set(self.pole_neighbors(node))
        return adj_list

    def bounds(self):
        min_x = min(p[0] for p in self.nodes)
        max_x = max(p[0] for p in self.nodes)
        min_y = min(p[1] for p in self.nodes)
        max_y = max(p[1] for p in self.nodes)
        return min_x, max_x, min_y, max_y

    def empty_spots(self, bounds=None) -> Generator[Pos, None, None]:
        min_x, max_x, min_y, max_y = bounds or self.bounds()
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) not in self.nodes:
                    yield x, y

    def all_pole_coverage(self, pole: Pos) -> Generator[Pos, None, None]:
        px, py = pole
        dd = (self.pole_coverage - 1) // 2
        for x in range(px - dd, px + dd + 1):
            for y in range(py - dd, py + dd + 1):
                yield x, y

    def all_of_type(self, node_type: int) -> set[Pos]:
        return {pt for pt, t in self.nodes.items() if t == node_type}

    def all_of_types(self, node_types: set[NodeType]) -> set[Pos]:
        return {pt for pt, t in self.nodes.items() if t in node_types}

    def powered_entities(self, pole: Pos) -> set[Pos]:
        return {
            pt for pt in self.all_pole_coverage(pole)
            if self.nodes.get(pt) == NodeType.Powerable
        }

    def candidate_possible_pole(
            self,
            pos: Pos,
            bounds: tuple[int, int, int, int] = None
    ) -> CandidatePole:
        return CandidatePole(pos, self.powered_entities(pos), set(self.possible_pole_neighbors(pos,
                                                                                               bounds=bounds
                                                                                               )))

    def candidate_existing_pole(self, pos: Pos) -> CandidatePole:
        return CandidatePole(pos, self.powered_entities(pos), set(self.pole_neighbors(pos)))

    def all_candidate_poles(self, ) -> list[CandidatePole]:
        bounds = self.bounds()
        return [
            self.candidate_possible_pole(pos, bounds)
            for pos in set(self.empty_spots(bounds)) | self.all_of_type(NodeType.Pole)
        ]

    def all_existing_poles(self) -> list[CandidatePole]:
        return [self.candidate_existing_pole(pos) for pos in self.all_of_type(NodeType.Pole)]

    def matplot(self, show: bool = True):
        for (x, y), node_type in self.nodes.items():
            if node_type == Pole:
                plt.plot(x, y, 'ro')
            elif node_type == Blocker:
                plt.plot(x, y, 'b.')
            else:
                plt.plot(x, y, 'gx')

        min_x, max_x, min_y, max_y = self.bounds()
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(min_y - 1, max_y + 1)
        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()
