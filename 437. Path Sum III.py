from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.numOfPaths = 0


class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.numOfPaths = 0
        self.dfs(root, targetSum)
        return self.numOfPaths

    def dfs(self, node, target):
        if node is None:
            return
        self.test(node, target)
        self.dfs(node.left, target)
        self.dfs(node.right, target)

    def test(self, node, target):
        if node is None:
            return
        if node.val == target:
            self.numOfPaths += 1

        self.test(node.left, target - node.val)
        self.test(node.right, target - node.val)


treeNode1 = TreeNode()
treeNode2 = TreeNode()
treeNode3 = TreeNode()
treeNode4 = TreeNode()
treeNode5 = TreeNode()
treeNode6 = TreeNode()
treeNode7 = TreeNode()
treeNode8 = TreeNode()
treeNode9 = TreeNode()
treeNode1.val = 10
treeNode2.val = 5
treeNode3.val = 3
treeNode4.val = 3
treeNode5.val = -2
treeNode6.val = 2
treeNode7.val = 1
treeNode8.val = -3
treeNode9.val = 11
treeNode1.left = treeNode2
treeNode1.right = treeNode8
treeNode2.left = treeNode3
treeNode2.right = treeNode6
treeNode3.left = treeNode4
treeNode3.right = treeNode5
treeNode6.right = treeNode7
treeNode8.right = treeNode9

slt = Solution()
print(slt.pathSum(treeNode1, 8))
