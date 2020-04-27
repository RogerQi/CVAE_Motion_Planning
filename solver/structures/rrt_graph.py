import numpy as np
import heapq
from .disjoint_set import disjoint_set

class rrt_multi_root_graph(object):
    '''
    A class of specially weighted undirected graph representing the constructed RRT graph
    constructed with multiple initial sources.
    '''
    def __init__(self, source, sink, metric):
        self.vertices = []
        self.edge_dict = {}
        self.v_to_dset_id = {}
        self.vertex_dset = disjoint_set()
        self.source = source
        self.sink = sink
        self.metric = metric
        self.add_v(self.source) # 0
        self.add_v(self.sink)   # 1
    
    def add_v(self, v):
        assert v not in self.vertices
        self.vertices.append(v)
        assigned_id = self.vertex_dset.add_one_and_get_id()
        self.v_to_dset_id[v] = assigned_id
        self.edge_dict[v] = {}
    
    def add_e(self, node_a, node_b):
        # For hash table purpose. We need to enforce some sort of total ordering
        # Here we use a heuristic: id of the ultimate root of node_a and node_b.
        v_a = node_a.ultimate_root
        v_b = node_b.ultimate_root
        new_edge_cost = node_a.local_cost + node_b.local_cost
        assert v_a in self.vertices and v_b in self.vertices
        try:
            cur_edge = self.edge_dict[v_a][v_b]
            cur_edge_cost = cur_edge[0].local_cost + cur_edge[1].local_cost
            if new_edge_cost >= cur_edge_cost:
                return # new edge is not efficient. Do nothing
        except KeyError:
            pass # edge not in graph yet. Add it!
        self.edge_dict[v_a][v_b] = (node_a, node_b)
        self.edge_dict[v_b][v_a] = (node_b, node_a)
        self.vertex_dset.union(self.v_to_dset_id[v_a], self.v_to_dset_id[v_b])
    
    def is_solved(self):
        return self.vertex_dset.find(0) == self.vertex_dset.find(1)

    def get_soln(self):
        if len(self.vertices) == 2:
            return self.get_soln_w_only_start_and_goal()
        else:
            return self.get_soln_w_astar()

    def get_soln_w_only_start_and_goal(self):
        soln_node_a, soln_node_b = self.edge_dict[self.source][self.sink]
        first_path = []
        second_path = []
        cur_node = soln_node_a
        while cur_node is not None:
            first_path.append(cur_node.state)
            cur_node = cur_node.parent
        first_path = first_path[::-1]
        cur_node = soln_node_b
        while cur_node is not None:
            second_path.append(cur_node.state)
            cur_node = cur_node.parent
        return first_path + second_path
    
    def get_soln_w_astar(self):
        '''
        Perform a A* search on solved graph to obtain a path from source to goal
        '''
        assert self.is_solved()
        def get_h(cur_state):
            return self.metric(cur_state, self.sink.state)
    
        class astar_node(object):
            def __init__(self, graph_node, parent_node, g):
                self.parent = parent_node
                self.graph_node = graph_node
                self.g = g
                self.h = get_h(self.graph_node.state)
                self.f = self.g + self.h

            def __lt__(self, other):
                return self.f < other.f

            def __gt__(self, other):
                return self.f > other.f

            def __eq__(self, other):
                return self.f == other.f

            def __le__(self, other):
                return (self < other) or (self == other)
            
            def __ge__(self, other):
                return (self > other) or (self == other)

            def __hash__(self):
                return hash(self.graph_node)

        start_node = astar_node(self.source, None, 0)
        frontier = [start_node]
        graph_node_visited_dict = {start_node.graph_node: True}
        while len(frontier) > 0:
            cur_node = heapq.heappop(frontier)
            # Test if goal
            if cur_node.graph_node == self.sink:
                # Back track for solution
                cur_bt_node = cur_node
                high_level_v_path = []
                while cur_bt_node is not None:
                    high_level_v_path.append(cur_bt_node.graph_node)
                    cur_bt_node = cur_bt_node.parent
                high_level_v_path = high_level_v_path[::-1]
                assert high_level_v_path[0] == self.source
                assert high_level_v_path[-1] == self.sink
                ret_state_path = []
                for i in range(len(high_level_v_path) - 1):
                    cur_v = high_level_v_path[i]
                    next_v = high_level_v_path[i + 1]
                    first_path = []
                    second_path = []
                    bridge_a, bridge_b = self.edge_dict[cur_v][next_v]
                    while bridge_a is not None:
                        first_path.append(bridge_a.state)
                        bridge_a = bridge_a.parent
                    first_path = first_path[::-1]
                    while bridge_b is not None:
                        second_path.append(bridge_b.state)
                        bridge_b = bridge_b.parent
                    ret_state_path += (first_path + second_path)
                return ret_state_path
            # Keep expanding
            for adj_graph_node in self.edge_dict[cur_node.graph_node].keys():
                if adj_graph_node in graph_node_visited_dict:
                    continue
                # Add
                bridge_node_a, bridge_node_b = self.edge_dict[cur_node.graph_node][adj_graph_node]
                edge_cost = bridge_node_a.local_cost + bridge_node_b.local_cost + self.metric(bridge_node_a.state, bridge_node_b.state)
                new_node = astar_node(adj_graph_node, cur_node, cur_node.g + edge_cost)
                graph_node_visited_dict[adj_graph_node] = True
                heapq.heappush(frontier, new_node)