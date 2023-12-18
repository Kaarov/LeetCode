from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root

        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)

        if l and r:
            return root
        return l or r


treeNode1 = TreeNode(3)
treeNode2 = TreeNode(5)
treeNode3 = TreeNode(6)
treeNode4 = TreeNode(2)
treeNode5 = TreeNode(7)
treeNode6 = TreeNode(4)
treeNode7 = TreeNode(1)
treeNode8 = TreeNode(0)
treeNode9 = TreeNode(8)
treeNode1.left = treeNode2
treeNode1.right = treeNode7
treeNode2.left = treeNode3
treeNode2.right = treeNode4
treeNode4.left = treeNode5
treeNode4.right = treeNode6
treeNode7.left = treeNode8
treeNode7.right = treeNode9


slt = Solution()
print(slt.lowestCommonAncestor(treeNode1, treeNode2, treeNode7))

