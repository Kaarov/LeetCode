from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None
        queue = [root]

        while queue:
            current = queue.pop(0)
            if current.val == val:
                queue = []
                return current
            if current.left: queue.append(current.left)
            if current.right: queue.append(current.right)

        return None

# Done ✅
