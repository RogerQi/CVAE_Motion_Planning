import sys
import os
import time
import math
import numpy as np
import klampt
from klampt.vis.ipython import KlamptWidget

def add_path(custom_path):
    if custom_path not in sys.path: sys.path.insert(0, custom_path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', '..', 'solver')
add_path(lib_path)

# import from added path
from base_world import base_world

world_config_file = "data/ur5Blocks.xml"

class arm_n_blocks_world(base_world):
    def __init__(self, random_plan = False):
        self.world = klampt.WorldModel()
        self.world.readFile(world_config_file)
        self.robot = self.world.robot(0)

        self.obstacles = []
        for i in range(self.world.numTerrains()):
            self.obstacles.append(self.world.terrain(i))

        self.initialize(random_plan)
        while not self.test(self.start_conf) or not self.test(self.goal_conf):
            self.initialize(random_plan)
    
    def initialize(self, random_plan):
        # Initialize start configuration
        if random_plan:
            raise NotImplementedError
        else:
            # Set the home configuration as in cs498 HW3
            self.start_conf = self.robot.getConfig()
        # Initialize goal configuration
        if random_plan:
            raise NotImplementedError
        else:
            self.goal_conf = self.robot.getConfig()
            self.goal_conf[1] = 2.3562
            self.goal_conf[2] = -math.pi/3.0
            self.goal_conf[3] = math.pi*2/3.0
            self.goal_conf[4] = 0.64181
            self.goal_conf[5] = math.pi/2.0

    def test(self, cur_conf):
        self.robot.setConfig(cur_conf)
        # Test self-collision
        if self.robot.selfCollides():
            return False
        # Test obstacle
        for i in range(self.robot.numLinks()):
            cur_link = self.robot.link(i)
            geo = cur_link.geometry()
            for o in self.obstacles:
                if i == 0 and o.getName() == "cart_support":
                    continue # The contact (or collision) between the robot arm base and its support is fine
                if geo.collides(o.geometry()):
                    return False
        return True

    def plot(self):
        '''
        Plotter

        Return:
            kvis: klampt kvis display handle
        '''
        try:
            assert get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except (NameError, AssertionError) as e:
            raise NotImplementedError("Currently, 3D plotting is supported only in jupyter notebook.")
        # Assume running in jupyter/qt console from this point.
        kvis = KlamptWidget(self.world, width=600, height=400)
        kvis.setCamera({u'near': 0.1,
            u'target':
                {u'y': 1.1188322004142854,
                u'x': 0.042176605196346695,
                u'z': 0.009329657831685366},
            u'far': 1000,
            u'position':
                {u'y': 1.6142988723996625,
                u'x': 1.5814619610767169,
                u'z': -0.03643442712963929},
            u'up': {u'y': 1, u'x': 0, u'z': 0}})

        display(kvis)
        return kvis