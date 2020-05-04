import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patches

import random
import numpy as np
import numpy.linalg as npla
from config import USE_GPU
if USE_GPU:
    import torch

class Circle(object):
    """A circle geometry.  Can collide with circles or rectangles."""
    def __init__(self, center, radius):
        assert isinstance(center, tuple) or center.shape == (2,)
        self.center = center
        self.radius = radius
        self.radius_squared = radius * radius
        self.dummy_obstacle = False

    def contains(self, p):
        return npla.norm(self.center - p) <= self.radius

    def robot_collides(self, robot_center, robot_radius):
        return npla.norm(self.center - robot_center) <= (self.radius + robot_radius)

    def draw_matplotlib(self, ax, **args):
        ax.add_patch(patches.Circle(self.center, self.radius, color = "k", **args))
    
    def get_parameter(self):
        return [self.center, self.radius]

class Robot(Circle):
    def __init__(self, center, radius, goal, id):
        super(Robot, self).__init__(center, radius)
        self.goal = goal
        self.id = id

    def draw_matplotlib(self, ax, **args):
        ax.add_patch(patches.Circle(self.center, self.radius, color = "red", **args))
        ax.text(self.center[0], self.center[1], str(self.id))
        ax.add_patch(patches.Circle(self.goal, self.radius, color = "blue", **args))
        ax.text(self.goal[0], self.goal[1], str(self.id))
    
    def robot_robot_collides(self, my_center, other_robot_center):
        return npla.norm(my_center - other_robot_center) <= (self.radius + self.radius)

class Rectangle(object):
    def __init__(self, bmin, bmax):
        bmin = np.array(bmin)
        bmax = np.array(bmax)
        assert bmin.shape == (2,)
        assert bmax.shape == (2,)
        # Standard CV coordinate
        assert bmin[0] <= bmax[0]
        assert bmin[1] <= bmax[1]
        if bmin[0] == bmax[0] or bmin[1] == bmax[1]:
            self.dummy_obstacle = True
        else:
            self.dummy_obstacle = False
        self.bmin = bmin
        self.bmax = bmax

    def contains(self, p):
        if self.dummy_obstacle: return False
        return (self.bmin[0] <= p[0] <= self.bmax[0]) and (self.bmin[1] <= p[1] <= self.bmax[1])

    def robot_collides(self, robot_center, robot_radius):
        if self.dummy_obstacle: return False
        closest = np.array([max(self.bmin[0], min(robot_center[0], self.bmax[0])),
                    max(self.bmin[1], min(robot_center[1], self.bmax[1]))])
        return npla.norm(robot_center - closest) <= robot_radius

    def draw_matplotlib(self, ax, **args):
        if self.dummy_obstacle: return
        ax.add_patch(patches.Rectangle(self.bmin, self.bmax[0]-self.bmin[0], self.bmax[1]-self.bmin[1], **args))
    
    def get_parameter(self):
        if self.dummy_obstacle: return []
        return [self.bmin, self.bmax]

class tilted_rect(object):
    def __init__(self):
        pass

def create_obstacles(num):
    obstacles_collection = []
    for i in range(num):
        obstacles = []
        num_obs = random.randint(1,10)
        for j in range(num_obs):
            if random.randint(0,1):
                obstacles.append(Rectangle([round(random.random(), 3),round(random.random(), 3)],[round(random.random(), 3),round(random.random(), 3)]))
            else:
                rand_center = (round(random.random(), 3), round(random.random(), 3))
                radius_limit = [rand_center[0], 1 - rand_center[0], rand_center[1], 1 - rand_center[1]]
                rand_radius = round(random.uniform(0, min(radius_limit)), 3)
                obstacles.append(Circle(rand_center,rand_radius))
        obstacles_collection.append(obstacles)
    return obstacles_collection

if __name__ == "__main__":
    obstacles = [Rectangle([0.3,0],[0.7,0.4]),Rectangle([0.3,0.6],[0.7,1.0])]
    plt.figure(figsize=(8,8))
    plt.axis('equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    for o in obstacles:
        o.draw_matplotlib(plt.gca(),color='k')
    plt.show()