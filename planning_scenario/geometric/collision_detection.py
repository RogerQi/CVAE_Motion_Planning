import numpy as np
import numpy.linalg as npla
from geometric_objects import Rectangle, Circle, Robot
from config import USE_GPU

# For simplicity, right now we use PyTorch for GPU computation.
# and numpy for vectorized CPU computation
if USE_GPU:
    import torch

def test_circle_collision():
    pass

class collision_detection_module(object):
    def __init__(self, robot_radius):
        self.robot_radius = robot_radius
        self.two_robot_radius = robot_radius + robot_radius
        self.robot_radius_squared = self.robot_radius * self.robot_radius
        self.two_robot_radius_squared = self.two_robot_radius * self.two_robot_radius

    def _register_obstacle(self, obstacle_list):
        self.circle_obstacles = []
        self.rect_lower_bound_vec = []
        self.rect_upper_bound_vec = []
        for o in obstacle_list:
            if isinstance(o, Circle):
                self.circle_obstacles.append((o.center, o.radius))
            elif isinstance(o, Rectangle):
                self.rect_lower_bound_vec.append(o.bmax)
                self.rect_upper_bound_vec.append(o.bmin)
            else:
                raise NotImplementedError("Unable to vectorize: unsupported geometric object.")
        self.num_circles = len(self.circle_obstacles)
        self.num_rects = len(self.rect_lower_bound_vec)
        # Pad into matrices
        # Circle
        # center_mat shape: (1 (to be boardcasted), n (to be boardcasted), 2)
        # radius_mat shape: (1, n, 1)
        self.center_mat = [a[0] for a in self.circle_obstacles]
        self.center_mat = np.array(self.center_mat).reshape((1, -1, 2))
        self.radius_mat = [a[1] for a in self.circle_obstacles]
        self.radius_mat = np.array(self.radius_mat).reshape((1, -1, 1))
        self.radius_squared_mat = np.square(self.radius_mat + self.robot_radius)
        # Rect
        self.rect_lower_bound_vec = np.array(self.rect_lower_bound_vec).reshape((1, -1, 2))
        self.rect_upper_bound_vec = np.array(self.rect_upper_bound_vec).reshape((1, -1, 2))

    def batch_feasibility_detect(self, input_tensor):
        '''
        Batch collision-detection

        Args:
            - input_tensor: tensor of dimension B x R x DOF, where
                * B is the number of configuration in the batch,
                * R is the number of agents in the world (i.e. num_robot)
                * DOF is the degree of freedom for each robot
        
        Return:
            ret_vec: tensor of dimension (B,), where ret_vec[i] is True if
                input_tensor[i] is feasible; false otherwise.
        '''
        assert len(input_tensor.shape) == 3
        B, R, DOF = input_tensor.shape
        if USE_GPU:
            raise NotImplementedError
            # Turn off autograd for SPEEEEEEEEEEED
            with torch.no_grad():
                pass
        else:
            ret_vec = np.ones((B,), dtype = np.bool)
            # Detect boundary
            bound_vec = (input_tensor > 0) & (input_tensor < 1)
            bound_condition = np.all(bound_vec, axis = (1, 2))
            ret_vec = ret_vec & bound_condition
            # Obstacle collision detection
            # Circle
            if self.num_circles > 0:
                # Underlying condition: npla.norm(self.center - robot_center) <= (self.radius + robot_radius)
                # TODO: this needs to be thoroughly checked
                robot_circle_lhs = np.repeat(input_tensor, repeats = self.num_circles, axis = 1)
                center_vec = np.repeat(self.center_mat, repeats = B, axis = 0)
                center_vec = np.repeat(center_vec, repeats = R, axis = 0).reshape((B, -1, DOF))
                circle_lhs_vec = np.square(robot_circle_lhs - center_vec).sum(axis = 2)
                circle_cond_vec = circle_lhs_vec > self.radius_squared_mat
                circle_cond_vec = np.all(circle_cond_vec, axis = 1)
                ret_vec = ret_vec & circle_cond_vec
            # Rect
            # Compute closet point
            # Underlying condition:
            # closest = np.array([max(self.bmin[0], min(robot_center[0], self.bmax[0])),
            # max(self.bmin[1], min(robot_center[1], self.bmax[1]))])
            # return npla.norm(robot_center - closest) <= robot_radius
            if self.num_rects > 0:
                robot_center_vec = np.repeat(input_tensor, repeats = self.num_rects, axis = 1)
                lower_bound_vec = np.repeat(self.rect_lower_bound_vec, repeats = B, axis = 0)
                lower_bound_vec = np.repeat(lower_bound_vec, repeats = R, axis = 0).reshape((B, -1, DOF))
                closet_pt_vec = np.minimum(robot_center_vec, lower_bound_vec)
                upper_bound_vec = np.repeat(self.rect_upper_bound_vec, repeats = B, axis = 0)
                upper_bound_vec = np.repeat(upper_bound_vec, repeats = R, axis = 0).reshape((B, -1, DOF))
                closet_pt_vec = np.maximum(closet_pt_vec, upper_bound_vec)
                diff_vec = robot_center_vec - closet_pt_vec
                diff_vec = np.square(diff_vec).sum(axis = 2)
                diff_vec = diff_vec > self.robot_radius_squared
                rect_cond_vec = np.all(diff_vec, axis = 1)
                ret_vec = ret_vec & rect_cond_vec
            # Robot collision detection
            if R > 1:
                for i in range(R - 1):
                    cur_robot_center = input_tensor[:,i:i+1,:]
                    cur_robot_center = np.repeat(cur_robot_center, repeats = R - i - 1, axis = 1) # Boardcast
                    other_robot_to_test = input_tensor[:,i+1:,]
                    robot_dist_vec = np.square(cur_robot_center - other_robot_to_test).sum(axis = 2)
                    robot_collision_cond = (robot_dist_vec > self.two_robot_radius_squared).flatten()
                    ret_vec = ret_vec & robot_collision_cond
            return ret_vec