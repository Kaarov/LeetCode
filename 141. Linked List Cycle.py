# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def hasCycle(self, head: ListNode | None) -> bool:
        visited_node = {}

        while head:
            if head in visited_node:
                return True

            visited_node[head] = True
            head = head.next

        return False


if __name__ == "__main__":
    node3 = ListNode(-4)
    node2 = ListNode(0)
    node1 = ListNode(2)
    head1 = ListNode(3)
    head1.next = node1
    node1.next = node2
    node2.next = node3
    node3.next = node1

    node1 = ListNode(2)
    head2 = ListNode(1)
    head2.next = node1

    slt = Solution()
    assert slt.hasCycle(head1) is True
    assert slt.hasCycle(head2) is False

# Done ✅
