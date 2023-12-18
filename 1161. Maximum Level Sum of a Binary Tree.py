# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def __init__(self):
        self.ans = {}

    def maxLevelSum(self, root: 'TreeNode') -> int:
        self.test(root, 1)

        return max(self.ans, key=self.ans.get)

    def test(self, root: 'TreeNode', level: int):
        if not root:
            return

        self.ans[level] = self.ans.get(level, 0) + root.val
        self.test(root.left, level + 1)
        self.test(root.right, level + 1)


treeNode1 = TreeNode()
treeNode2 = TreeNode()
treeNode3 = TreeNode()
treeNode4 = TreeNode()
treeNode5 = TreeNode()
treeNode1.val = 1
treeNode2.val = 7
treeNode3.val = 7
treeNode4.val = -8
treeNode5.val = 0
treeNode1.left = treeNode2
treeNode1.right = treeNode5
treeNode2.left = treeNode3
treeNode2.right = treeNode4

slt = Solution()
print(slt.maxLevelSum(treeNode1))

# Done ✅
