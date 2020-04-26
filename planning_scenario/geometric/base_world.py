import numpy as np
import matplotlib.pyplot as plt
from config import ROBOT_RADIUS

import solver_wrapper

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
        assert solver in ["rrt", "bidirectional_rrt", "prm", "astar", "fmt"]
        if solver == "astar":
            ret = solver_wrapper.astar_solve(self)
            if ret is None:
                print("No solution found!")
        elif solver == "rrt":
            ret = solver_wrapper.rrt_solve(self)
            if ret is None:
                print("No solution found!")
        elif solver == "bidirectional_rrt":
            ret = solver_wrapper.bidirectional_rrt_solve(self)
            if ret is None:
                print("No solution found!")
        else:
            raise NotImplementedError
        self.soln_dict[solver] = ret
        return ret

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
        try:
            ret = self.soln_dict['astar']
            return ret
        except KeyError:
            pass
        try:
            ret = self.soln_dict['rrt']
            return ret
        except KeyError:
            pass
        try:
            ret = self.soln_dict['bidirectional_rrt']
            return ret
        except KeyError:
            pass
        raise NotImplementedError