class disjoint_set(object):
    '''
    Implementation of the classical disjoint set in Python
    '''
    def __init__(self):
        self.dset = []

    def add_one_and_get_id(self):
        self.dset.append(-1)
        return len(self.dset) - 1
    
    def find(self, id):
        if self.dset[id] < 0:
            # base case
            return id
        self.dset[id] = self.find(self.dset[id]) # recurse and flatten
        return self.dset[id]
    
    def union(self, id_a, id_b):
        first_root = self.find(id_a)
        second_root = self.find(id_b)
        if first_root == second_root:
            return # nothing to be done
        if self.dset[first_root] < self.dset[second_root]:
            # first root contains more elements
            # merge second set into the first one
            self.dset[first_root] += self.dset[second_root]
            self.dset[second_root] = first_root
        else:
            # merge first set into second one
            self.dset[second_root] += self.dset[first_root]
            self.dset[first_root] = second_root

    def __len__(self):
        return len(self.dset)

if __name__ == '__main__':
    # a simple test
    test_ds = disjoint_set()
    test_ds.add_one_and_get_id() # 0
    test_ds.add_one_and_get_id() # 1
    test_ds.add_one_and_get_id() # 2
    test_ds.add_one_and_get_id() # 3
    test_ds.add_one_and_get_id() # 4
    for i in range(5):
        assert test_ds.find(i) == test_ds.find(i)
        for j in range(i + 1, 5):
            assert test_ds.find(i) != test_ds.find(j)
    test_ds.union(0, 1)
    test_ds.union(1, 2)
    test_ds.union(3, 4)
    assert test_ds.find(0) == test_ds.find(2)
    assert test_ds.find(2) != test_ds.find(3)
    assert test_ds.find(3) == test_ds.find(4)
    print("All tests passed!")