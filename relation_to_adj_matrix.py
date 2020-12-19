from __future__ import print_function
import numpy as np

"""
Basic operations on dependency trees.
"""


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = sorted(head, key=lambda x: x[2])
    head = [w[1] for w in head]
    # print(head, len(head))
    # tokens = tokens[:len(head)]
    # head = head
    root = None
    # print('head:'. head.size())
    # print('tokens:', )

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):

        h = head[i]
        # print('1111', h)
        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h-1].add_child(nodes[i])


    assert root is not None
    return root


def tree_to_adj(sent_len, tree, sent, not_directed=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    # ret = np.ones((sent_len, sent_len), dtype=np.float32)
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    length = ret.shape[0]

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    # if sent == 'sent2':
    #     for i in range(length):
    #         ret[length-1, i] = 1
    # elif sent == 'sent3':
    #     for i in range(length):
    #         ret[length-2, i] = 1
    #         ret[length-1, i] = 1
    # elif sent == 'sent4':
    #     for i in range(length):
    #         ret[length-3, i] = 1
    #         ret[length-2, i] = 1
    #         ret[length-1, i] = 1
    #
    if not_directed:
        ret = ret + ret.T

    ret = ret + np.eye(sent_len)

    return ret