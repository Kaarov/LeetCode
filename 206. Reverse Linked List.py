from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None

        while head:
            next = head.next
            head.next = prev
            prev = head
            head = next

        return prev


def is_equal(l1: ListNode, l2: ListNode) -> bool:
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1 = l1.next
        l2 = l2.next
    return l1 is None and l2 is None


if __name__ == "__main__":
    slt = Solution()

    node1 = ListNode(2, None)
    head1 = ListNode(1, node1)

    node1 = ListNode(1, None)
    expected_head1 = ListNode(2, node1)

    node4 = ListNode(5, None)
    node3 = ListNode(4, node4)
    node2 = ListNode(3, node3)
    node1 = ListNode(2, node2)
    head2 = ListNode(1, node1)

    node4 = ListNode(1, None)
    node3 = ListNode(2, node4)
    node2 = ListNode(3, node3)
    node1 = ListNode(4, node2)
    expected_head2 = ListNode(5, node1)

    assert is_equal(slt.reverseList(head1), expected_head1)
    assert is_equal(slt.reverseList(head2), expected_head2)

# Done ✅
