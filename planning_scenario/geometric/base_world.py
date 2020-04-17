import numpy as np
import matplotlib.pyplot as plt
import geometric_objects as gobj
import config

from solver import astar

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

    def plot(self, draw_ogrid = True, soln = None):
        fig = plt.figure()
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
        plt.show()

    def solve(self, solver):
        '''
        Return solution using specified solver
        '''
        assert solver in ["rrt", "prm", "astar", "fmt"]
        if solver == "astar":
            ret = astar.astar_solve(self)
            if ret is None:
                print("No solution found!")
        self.soln_dict[solver] = ret
        return ret

    def get_trainable_data(self):
        raise NotImplementedError

    def draw_occupany_grid(self, ax, num_samples):
        def test_pseudo_robot(cur_conf):
            for o in self.obstacles:
                if o.robot_collides(cur_conf, config.ROBOT_RADIUS):
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
        raise NotImplementedError