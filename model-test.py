import random
from copy import deepcopy
from math import ceil
from typing import Callable

import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult

from mst import mst, plot_adj_list
from pole_cover import solve_approx_pole_cover, get_center, \
    euclid_dist, get_pole_dists, dist_estimate1
from pole_grid import PoleGrid, Pos, NonPole, Pole, CandidatePole, small_pole
from set_cover import solve_set_cover


# use euclidean distance as the edge weight


def fill_random(
        grid: PoleGrid,
        blocker_density: float,
        powerable_density: float,
        size: int,
        center: Pos = (0, 0)
):
    all_pts = {(x, y) for x in range(size) for y in range(size)}
    num_pts = size * size
    blockers = random.sample(list(all_pts), ceil(num_pts * blocker_density))
    all_pts.difference_update(blockers)
    powerables = random.sample(list(all_pts), ceil(num_pts * powerable_density))

    for x, y in blockers:
        pt2 = center[0] + x - size // 2, center[1] + y - size // 2
        grid.add(NonPole.at_tile(pt2, False))

    for pt in powerables:
        pt2 = center[0] + pt[0] - size // 2, center[1] + pt[1] - size // 2
        grid.add(NonPole.at_tile(pt2, True))

    return grid


def main():
    g = PoleGrid()
    fill_random(g, 0.5, 0.15, 10, (-8, -8))
    fill_random(g, 0.5, 0.15, 12, (8, -10))
    fill_random(g, 0.5, 0.15, 12, (8, 8))
    fill_random(g, 0.5, 0.15, 12, (-10, 8))
    # g.add(NonPole.at_tile((0, 0), True))
    # g.add(NonPole.at_tile((-10, 0), True))
    # g.add(NonPole.at_tile((10, 0), True))

    # g.matplot()

    pole_to_use = small_pole
    possible_poles = g.all_candidate_poles(pole_to_use)
    center = get_center([pole.pos for pole in possible_poles])
    for pole in possible_poles:
        pole.cost = euclid_dist(pole.pos, center) / 1000 + 1

    dist_fun = dist_estimate1(pole_to_use)


    def try_with_solver(
            name: str,
            solver: Callable[[list[CandidatePole]], tuple[list[CandidatePole], OptimizeResult]],
    ):
        poles, res = solver(possible_poles)

        g2 = deepcopy(g)

        # ugly, but works for now
        for pole in poles:
            g2.add(Pole(pole.pos, pole_to_use.name))

        g2.matplot(show=False)
        plt.suptitle(name)
        plt.title(f"Num poles: {round(sum(res.x))}")

        adj_list = g2.pole_adj_list()
        the_mst = mst(adj_list)
        plot_adj_list(the_mst)
        plt.show()

        # # plot heatmap, using dists
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.plot(center[0] + 0.5, center[1] + 0.5, 'rX')
        # plt.scatter([pole.pos[0] for pole in possible_poles], [pole.pos[1] for pole in possible_poles],
        #             c=[dists[pole] for pole in possible_poles],
        #             marker='.'
        #             )
        # plt.colorbar()
        # plt.show()

    try_with_solver(
        "Set cover",
        lambda coverages: solve_set_cover(coverages)
    )

    try_with_solver(
        "Approx pole cover",
        lambda coverages: solve_approx_pole_cover(coverages, dist_fun, milp_options={
            "disp": True,
            "time_limit": 300,
        }))


if __name__ == "__main__":
    main()
