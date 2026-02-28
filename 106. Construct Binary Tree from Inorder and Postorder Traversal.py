# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, inorder: list[int], postorder: list[int]) -> TreeNode | None:
        if not inorder or not postorder:
            return None
        root = TreeNode(postorder[-1])
        index = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:index], postorder[:index])
        root.right = self.buildTree(inorder[index + 1:], postorder[index:-1])
        return root


def trees_equal(t1: TreeNode | None, t2: TreeNode | None) -> bool:
    if t1 is None and t2 is None:
        return True
    if t1 is None or t2 is None or t1.val != t2.val:
        return False
    return trees_equal(t1.left, t2.left) and trees_equal(t1.right, t2.right)


if __name__ == "__main__":
    slt = Solution()
    expected = TreeNode(3)
    expected.left = TreeNode(9)
    expected.right = TreeNode(20)
    expected.right.left = TreeNode(15)
    expected.right.right = TreeNode(7)

    result = slt.buildTree([9, 3, 15, 20, 7], [9, 15, 7, 20, 3])
    assert trees_equal(result, expected)
