from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.maxLength = 0


class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.maxLength = 0

        def solve(node, deep, dir):
            self.maxLength = max(self.maxLength, deep)

            if node.left is not None:
                solve(node.left, deep + 1, 'left') if dir != 'left' else solve(node.left, 1, 'left')
            if node.right is not None:
                solve(node.right, deep + 1, 'right') if dir != 'right' else solve(node.right, 1, 'right')

        solve(root, 0, '')
        return self.maxLength


treeNode1 = TreeNode()
treeNode2 = TreeNode()
treeNode3 = TreeNode()
treeNode4 = TreeNode()
treeNode5 = TreeNode()
treeNode6 = TreeNode()
treeNode7 = TreeNode()
treeNode8 = TreeNode()
treeNode1.val = 1
treeNode2.val = 1
treeNode3.val = 1
treeNode4.val = 1
treeNode5.val = 1
treeNode6.val = 1
treeNode7.val = 1
treeNode8.val = 1

treeNode1.right = treeNode2
treeNode2.left = treeNode3
treeNode2.right = treeNode4
treeNode4.left = treeNode5
treeNode4.right = treeNode6
treeNode5.right = treeNode7
treeNode7.right = treeNode8

slt = Solution()
print(slt.longestZigZag(treeNode1))
