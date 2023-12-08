from typing import Optional
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        queue = []
        if root is None:
            return []

        if root.left is None and root.right is None:
            return [root.val]

        ans = []
        queue.append(root)
        while queue:
            child_queue = []
            prev = -1
            while queue:
                curr = queue.pop(0)

                if curr.left is not None:
                    child_queue.append(curr.left)

                if curr.right is not None:
                    child_queue.append(curr.right)

                prev = curr

            ans.append(prev.val)
            queue = child_queue

        return ans
