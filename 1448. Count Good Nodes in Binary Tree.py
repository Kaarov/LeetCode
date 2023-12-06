# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, val):
            if not node:
                return 0
            elif node.val >= val:
                return 1 + dfs(node.left, node.val) + dfs(node.right, node.val)
            else:
                return dfs(node.left, val) + dfs(node.right, val)

        return dfs(root, root.val)

        # def solve(root, val):
        #     if root:
        #         k = solve(root.left, max(val, root.val)) + solve(root.right, max(val, root.val))
        #         if root.val >= val:
        #             k += 1
        #         return k
        #     return 0
        #
        # return solve(root, root.val)
