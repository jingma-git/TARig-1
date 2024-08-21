class Node(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos


class TreeNode(Node):
    def __init__(self, name, pos, parent=None):
        super(TreeNode, self).__init__(name, pos)
        self.children = []
        self.parent = parent
