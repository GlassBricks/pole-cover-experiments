import random
from copy import deepcopy
from math import ceil

from mst import mst, plot_adj_list
from pole_cover import *
from pole_grid import Pos, NonPole, Pole, small_pole


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
    cand_poles = g.all_candidate_poles(pole_to_use)
    center = get_center([pole.pos for pole in cand_poles])
    for pole in cand_poles:
        pole.cost = euclid_dist(pole.pos, center) / 1000 + 1

    dist_fun = dist_estimate1()

    def visualize_poles(poles: list[CandidatePole],
                        title: str,
                        with_mst=False):
        g2 = deepcopy(g)
        for pole in poles:
            g2.add(Pole(pole.pos, pole_to_use.name))

        g2.matplot(show=False)
        plt.suptitle(title)
        plt.title(f"Num poles: {len(poles)}")
        if with_mst:
            adj_list = g2.pole_adj_list()
            the_mst = mst(adj_list)
            plot_adj_list(the_mst)
        plt.show()

    poles, res = solve_set_cover(coverages=cand_poles, remove_subsets=True)
    visualize_poles(poles, "Set cover", with_mst=True)

    poles1, res1 = solve_approx_pole_cover(cand_poles, dist_fun, remove_equiv=True, milp_options={
        "time_limit": 15,
        "disp": True,
    })
    visualize_poles(poles1, "Approx pole cover", with_mst=True)

    # def do_full_steiner_tree(
    #         poles: list[CandidatePole],
    # ):
    #      poles = remove_subset_poles(poles, keep_empty=True)
    #      poles = [
    #          pole for pole in poles
    #          if pole.powered_entities or (abs(pole.pos[0] + 0.5) % 2 == 0) and (abs(pole.pos[1] + 0.5) % 2 == 0) and (
    #                  max(abs(pole.pos[0]), abs(pole.pos[1])) < 10
    #          )]
    #      remove_unreferenced_poles(set(poles))
    #     visualize_poles(poles, "candidate poles")
    #     starts = get_root_poles(poles)
    #     poles, res = solve_pole_cover_steiner_flow(poles, starts, options={
    #         "time_limit": 60*1,
    #         "disp": True,
    #         "parallel": True,
    #     })
    #     visualize_poles(poles, "steiner tree", with_mst=True)
    # do_full_steiner_tree(cand_poles)


if __name__ == "__main__":
    main()
