from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def zigZag(self, root: Optional[TreeNode], flag: bool):
        stack = [root]
        ans = 0
        while stack:
            current = stack.pop()
            if flag:
                if current.left:
                    stack.append(current.left)
                    ans += 1
            else:
                if current.right:
                    stack.append(current.right)
                    ans += 1
            flag = not flag
        return ans

    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        stack = [root]
        ans = 0

        while stack:
            current = stack.pop()
            ans = max(ans, self.zigZag(current, True), self.zigZag(current, False))
            if current.left: stack.append(current.left)
            if current.right: stack.append(current.right)

        return ans


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
