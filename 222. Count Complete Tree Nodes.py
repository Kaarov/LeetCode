from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        nodes = [root]
        ans = 0
        while nodes:
            node = nodes.pop(0)
            if node:
                ans += 1
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
        return ans


if __name__ == '__main__':
    node5 = TreeNode(6)
    node4 = TreeNode(5)
    node3 = TreeNode(4)
    node2 = TreeNode(3, left=node5)
    node1 = TreeNode(2, left=node3, right=node4)
    root = TreeNode(1, left=node1, right=node2)
    slt = Solution()
    print(slt.countNodes(root))

# Done ✅
