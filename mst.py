from _heapq import heappop, heappush
from collections import defaultdict
from typing import Any, NamedTuple

from matplotlib import pyplot as plt

from pole_cover import euclid_dist_squared

AdjList = dict[Any, set[Any]]


class QE(NamedTuple):
    d: float
    a: Any
    b: Any

    def __lt__(self, other):
        return self.d < other.d

    def __iter__(self):
        return iter((self.d, self.a, self.b))


def mst(adj_list: AdjList) -> AdjList:
    pts = set(adj_list.keys())
    res = defaultdict(set)
    start = next(iter(pts))
    visited = set()
    heap = [QE(0, start, start)]
    while heap:
        weight, node, parent = heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node != parent:
            res[parent].add(node)
        p1 = node.pos
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                p2 = neighbor.pos
                heappush(heap, QE(euclid_dist_squared(p1, p2), neighbor, node))
    return res


def plot_adj_list(adj_list: AdjList):
    for pt, neighbors in adj_list.items():
        # directed
        for neigh in neighbors:
            x1, y1 = pt.pos
            x2, y2 = neigh.pos
            dx = x2 - x1
            dy = y2 - y1
            plt.arrow(
                x1, y1, dx, dy,
                head_width=0.5, length_includes_head=True,
                alpha=0.6, zorder=10,
            )
