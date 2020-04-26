import numpy as np
from .disjoint_set import disjoint_set

class rrt_multi_root_graph(object):
    '''
    A class of specially weighted undirected graph representing the constructed RRT graph
    constructed with multiple initial sources.
    '''
    def __init__(self, source, sink):
        self.vertices = []
        self.edge_dict = {}
        self.v_to_dset_id = {}
        self.vertex_dset = disjoint_set()
        self.source = source
        self.sink = sink
        self.add_v(self.source) # 0
        self.add_v(self.sink)   # 1
    
    def add_v(self, v):
        assert v not in self.vertices
        self.vertices.append(v)
        assigned_id = self.vertex_dset.add_one_and_get_id()
        self.v_to_dset_id[v] = assigned_id
    
    def add_e(self, node_a, node_b):
        # For hash table purpose. We need to enforce some sort of total ordering
        # Here we use a heuristic: id of the ultimate root of node_a and node_b.
        if node_a.ultimate_root.id > node_b.ultimate_root.id:
            node_a, node_b = node_b, node_a # swap
        v_a = node_a.ultimate_root
        v_b = node_b.ultimate_root
        new_edge_cost = node_a.local_cost + node_b.local_cost
        assert v_a in self.vertices and v_b in self.vertices
        try:
            cur_edge = self.edge_dict[(v_a, v_b)]
            cur_edge_cost = cur_edge[0].local_cost + cur_edge[1].local_cost
            if new_edge_cost >= cur_edge_cost:
                return # new edge is not efficient. Do nothing
        except KeyError:
            pass # edge not in graph yet. Add it!
        self.edge_dict[(v_a, v_b)] = (node_a, node_b)
        self.vertex_dset.union(self.v_to_dset_id[v_a], self.v_to_dset_id[v_b])
    
    def is_solved(self):
        return self.vertex_dset.find(0) == self.vertex_dset.find(1)
    
    def get_soln(self):
        # TODO: now it's hardcoded for naive bidirectional RRT for testing purpose
        # i.e. the graph will contain only two vertices with one and only one edge.
        try:
            self.edge_dict[(self.sink, self.source)] # This should trigger KeyError due to total ordering
            assert False
        except KeyError:
            pass
        soln_node_a, soln_node_b = self.edge_dict[(self.source, self.sink)]
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