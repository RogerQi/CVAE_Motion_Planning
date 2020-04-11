import random
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

import config
import geometric_objects as gobj

class world(object):
    '''
    World that organizes everything for geometric planning problem.
    '''
    def __init__(self, num_robots, max_num_obstacles, h = 1.0, w = 1.0):
        assert h == 1.0 and w == 1.0 # should be scale to 1s
        self.obstacles = []
        self.robots = []
        self.initialize(num_robots, max_num_obstacles)

    def initialize(self, num_robots, max_num_obstacles):
        # Initialize obstacles
        num_obs = random.randint(1, max_num_obstacles)
        for j in range(num_obs):
            if random.randint(0,1):
                self.obstacles.append(gobj.Rectangle((round(random.random(), 3),round(random.random(), 3)),
                                        (round(random.random(), 3),round(random.random(), 3))))
            else:
                rand_center = (round(random.random(), 3), round(random.random(), 3))
                radius_limit = [rand_center[0], 1 - rand_center[0], rand_center[1], 1 - rand_center[1]]
                rand_radius = round(random.uniform(0, min(radius_limit)), 3)
                self.obstacles.append(gobj.Circle(rand_center,rand_radius))
        # Initialize start state and goal state
        for i in range(num_robots):
            start_pt = np.random.random(size = (2,))
            goal_pt = np.random.random(size = (2,))
            # TODO: use solver to ensure this selection of start_pt and goal_pt is good.
            self.robots.append(gobj.Robot(start_pt, config.ROBOT_RADIUS, goal_pt, i))

    def discretize(self):
        pass

    def plot(self):
        '''
        Use matplotlib to plot current world map for debugging purpose.
        '''
        plt.figure(figsize=(5, 5))
        plt.axis('equal')
        plt.xlim(0,1)
        plt.ylim(0,1)
        for o in self.obstacles:
            o.draw_matplotlib(plt.gca())
        for r in self.robots:
            r.draw_matplotlib(plt.gca())
        plt.show()

    def solve(self, solver):
        '''
        Return solution using specified solver
        '''
        assert solver in ["rrt", "prm", "astar", "fmt"]

if __name__ == '__main__':
    test_world = world(1, 10)
    test_world.plot()
