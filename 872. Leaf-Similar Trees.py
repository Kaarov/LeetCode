from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(root: Optional[TreeNode]) -> List[int]:
            ans = []
            stack = [root]

            while stack:
                current = stack.pop()
                if not current.left and not current.right:
                    ans.append(current.val)
                if current.right: stack.append(current.right)
                if current.left: stack.append(current.left)

            return ans

        return dfs(root1) == dfs(root2)

# Done ✅
