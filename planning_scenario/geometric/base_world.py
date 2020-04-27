import numpy as np
import matplotlib.pyplot as plt
from config import ROBOT_RADIUS

import solver_wrapper

# Solver look-up table sorted in terms of general quality of computed paths.
solver_lut = [
    ('astar', solver_wrapper.astar_solve),
    ('bidirectional_rrt_star', solver_wrapper.bidirectional_rrt_star_solve),
    ('bidirectional_rrt', solver_wrapper.bidirectional_rrt_solve),
    ('rrt', solver_wrapper.rrt_solve)
]

class base_geometric_world(object):
    def __init__(self):
        self.obstacles = [] # instances of geometric objects.
        self.robots = []    # instances of gobj.robot
        self.soln_dict = {}
        raise NotImplementedError

    def test_one(self, single_conf):
        raise NotImplementedError

    def test(self, robot_conf):
        raise NotImplementedError

    def plot(self, _ax = None, draw_ogrid = True, soln = None):
        ax = _ax
        if _ax is None:
            fig = plt.figure(figsize = (6, 6))
            ax = fig.add_subplot(111, aspect = 'equal')
        for o in self.obstacles:
            o.draw_matplotlib(ax, alpha = 0.6)
        for r in self.robots:
            r.draw_matplotlib(ax)
        if draw_ogrid:
            self.draw_occupany_grid(ax, 20)
        if soln is not None:
            soln = np.array(soln).reshape((-1, self.num_robots, 2))
            for i in range(self.num_robots):
                ax.plot(soln[:,i,0], soln[:,i,1])
        if _ax is None:
            plt.show()

    def solve(self, solver):
        '''
        Return solution using specified solver
        '''
        for soln_name, soln_func in solver_lut:
            if solver == soln_name:
                ret = soln_func(self)
                if ret is None:
                    print("No solution found!")
                self.soln_dict[solver] = ret
                return ret
        raise NotImplementedError("Called with unimplemented solver {0}".format(solver))

    def get_trainable_data(self):
        raise NotImplementedError

    def draw_occupany_grid(self, ax, num_samples):
        def test_pseudo_robot(cur_conf):
            for o in self.obstacles:
                if o.robot_collides(cur_conf, ROBOT_RADIUS):
                    return False
            return True
        x_sample = np.linspace(0, 1, num_samples)
        y_sample = np.linspace(0, 1, num_samples)
        for x in x_sample:
            for y in y_sample:
                if test_pseudo_robot(np.array((x, y))):
                    # draw a green dot
                    ax.scatter(x, y, color = "green", s = 70, alpha = 0.8)
                else:
                    # draw a red dot
                    ax.scatter(x, y, color = "red", s = 70, alpha = 0.8)

    def get_best_soln(self):
        for soln_name, _ in solver_lut:
            try:
                return self.soln_dict[soln_name]
            except KeyError:
                pass
        raise NotImplementedError("Called with unimplemented solver {0}".format(solver))