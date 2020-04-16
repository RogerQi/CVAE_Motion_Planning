import numpy as np

class base_world(object):
    def __init__(self):
        raise NotImplementedError

    def test(self, robot_conf):
        raise NotImplementedError

    def plot(self, soln = None):
        raise NotImplementedError

    def solve(self, solver):
        raise NotImplementedError

    def get_trainable_data(self):
        raise NotImplementedError

    def get_best_soln(self):
        try:
            return self.soln_dict['astar']
        except KeyError:
            pass
        raise NotImplementedError