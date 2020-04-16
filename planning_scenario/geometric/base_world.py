import numpy as np
import geometric_objects as gobj
import config

class base_geometric_world(object):
    def __init__(self):
        self.obstacles = [] # instances of geometric objects.
        self.robots = []    # instances of gobj.robot
        raise NotImplementedError

    def test_one(self, single_conf):
        raise NotImplementedError

    def test(self, robot_conf):
        raise NotImplementedError

    def plot(self, draw_ogrid = True, soln = None):
        raise NotImplementedError

    def solve(self, solver):
        raise NotImplementedError

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
            return self.soln_dict['astar']
        except KeyError:
            pass
        raise NotImplementedError