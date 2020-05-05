import matplotlib as mpl
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

    def tilted_rect_robot_collides(self, tilted_rect_pt_set):
        '''
        Test if any axis can be used as a separating axis. If none of the axis can be used, then objects collide.
        '''
        my_pt_set = (self.bmin, (self.bmax[0], self.bmin[1]), self.bmax, (self.bmin[0], self.bmax[1]))
        # My separating axis
        for cur_pt_set in (my_pt_set, tilted_rect_pt_set):
            for i in range(4):
                pt_a = cur_pt_set[i]
                pt_b = cur_pt_set[(i + 1) % 4]
                normal_x = pt_b[1] - pt_a[1]
                normal_y = pt_a[0] - pt_b[0]
                # self.rect
                min_a = None
                max_a = None
                for p in my_pt_set:
                    projected = normal_x * p[0] + normal_y * p[1]
                    if min_a is None or projected < min_a:
                        min_a = projected
                    if max_a is None or projected > max_a:
                        max_a = projected
                # Tilted Rect
                min_b = None
                max_b = None
                for p in tilted_rect_pt_set:
                    projected = normal_x * p[0] + normal_y * p[1]
                    if min_b is None or projected < min_b:
                        min_b = projected
                    if max_b is None or projected > max_b:
                        max_b = projected
                if (max_a < min_b) or (max_b < min_a):
                    return False # no intersection
        return True

    def draw_matplotlib(self, ax, **args):
        if self.dummy_obstacle: return
        ax.add_patch(patches.Rectangle(self.bmin, self.bmax[0]-self.bmin[0], self.bmax[1]-self.bmin[1], **args))
    
    def get_parameter(self):
        if self.dummy_obstacle: return []
        return [self.bmin, self.bmax]

class tilted_rect_robot(object):
    def __init__(self, center, width, length, theta):
        self.center = center
        self.width = width
        self.length = length
        self.theta = theta

    def draw_matplotlib(self, ax, **args):
        bottom_left_pt = (self.center[0] - self.length / 2, self.center[1] - self.width / 2)

        t_start = ax.transData
        coords = t_start.transform([self.center[0], self.center[1]])
        t = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], self.theta)
        t_end = t_start + t

        rect = patches.Rectangle(bottom_left_pt, self.length, self.width, color = "red", **args)
        rect.set_transform(t_end)

        ax.add_patch(rect)
    
    @staticmethod
    def get_pt_set(center, width, length, theta):
        cos_theta = np.cos(theta) # Cached trigonometry value
        sin_theta = np.sin(theta)
        w_half_x_offset = cos_theta * width * 0.5 #Ox cos
        w_half_y_offset = sin_theta * width * 0.5 #Ox sin
        l_half_x_offset = cos_theta * length * 0.5 #Oy sin
        l_half_y_offset = sin_theta * length * 0.5 #Oy cos
        # points
        pt_a = (center[0] - l_half_x_offset + w_half_x_offset, center[1] - l_half_y_offset - w_half_y_offset)
        pt_b = (center[0] + l_half_x_offset + w_half_x_offset, center[1] + l_half_y_offset - w_half_y_offset)
        pt_c = (center[0] + l_half_x_offset - w_half_x_offset, center[1] + l_half_y_offset + w_half_y_offset)
        pt_d = (center[0] - l_half_x_offset - w_half_x_offset, center[1] - l_half_y_offset + w_half_y_offset)
        return (pt_a, pt_b, pt_c, pt_d)

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
    obstacles = [Rectangle([0.2, 0.2],[0.25,0.8])]
    tilted_rect = tilted_rect_robot((0.27809911, 0.5359022), 0.03, 0.15, 3.1131204936180885)
    print(tilted_rect)
    tilted_pt_set = tilted_rect_robot.get_pt_set(tilted_rect.center, tilted_rect.width, tilted_rect.length, tilted_rect.theta)
    print("my point set: {}".format(tilted_pt_set))
    res = obstacles[0].tilted_rect_robot_collides(tilted_pt_set)
    print("Collision or not: {}".format(res))
    plt.figure(figsize=(8,8))
    plt.axis('equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    for o in obstacles:
        o.draw_matplotlib(plt.gca(),color = 'blue', alpha = 0.4)
    tilted_rect.draw_matplotlib(plt.gca(), alpha = 0.6)
    plt.show()