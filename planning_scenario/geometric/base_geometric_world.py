import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from config import ROBOT_RADIUS
from matplotlib import animation

def add_path(custom_path):
    if custom_path not in sys.path: sys.path.insert(0, custom_path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', '..', 'solver')
add_path(lib_path)

# import from added path
from base_world import base_world
import solver_wrapper

# Solver look-up table sorted in terms of general quality of computed paths.
solver_lut = [
    ('A*', solver_wrapper.astar_solve),
    ('bRRT*', solver_wrapper.bidirectional_rrt_star_solve),
    ('FMT*', solver_wrapper.fmt_star_solve),
    ('bRRT', solver_wrapper.bidirectional_rrt_solve),
    ('RRT', solver_wrapper.rrt_solve)
]

class base_geometric_world(base_world):
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

    def anim(self, _ax = None, draw_ogrid = True, soln = None):
        ax = _ax
        if _ax is None:
            fig = plt.figure(figsize = (6, 6))
            ax = fig.add_subplot(111, aspect = 'equal')
        for o in self.obstacles:
            o.draw_matplotlib(ax, alpha = 0.6)
        # for r in self.robots:
        #     r.draw_matplotlib(ax)
        if draw_ogrid:
            self.draw_occupany_grid(ax, 20)
        # for i in range(self.num_robots):
        #     ax.plot(soln[:,i,0], soln[:,i,1])
        if soln is not None:
            soln = np.array(soln).reshape((-1, self.num_robots, 2))
            patches = []
            for j in range(self.num_robots):
                patch = self.robots[j].draw_matplotlib(ax)
                patches.append(patch)

            def anim_init():
                for patch in patches:
                    ax.add_patch(patch)
                return []

            def anim_manage(i):
                for j in range(self.num_robots):
                    anim_animate(i, j, patches[j])
                return []

            def anim_animate(i, j, patch):
                step = int(np.floor(i/100))
                if step == 0:
                    start = self.robots[j].ori
                    end = soln[step+1,j]
                    patch.center = (start[0] + (end[0]-start[0])*(i-step*100)/100, start[1] + (end[1]-start[1])*(i-step*100)/100)
                elif step < len(soln)-1:
                    start = soln[step,j]
                    end = soln[step+1,j]
                    patch.center = (start[0] + (end[0]-start[0])*(i-step*100)/100, start[1] + (end[1]-start[1])*(i-step*100)/100)
                return patch,

            anim = animation.FuncAnimation(fig, anim_manage, init_func=anim_init, frames=1500, interval=1, blit=True, repeat=False)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=200)
            anim.save('bRRT_1.mp4', writer=writer)

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
