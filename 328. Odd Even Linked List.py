from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: return None

        odd = head
        evenHead = even = head.next

        while even and even.next:
            odd.next = odd.next.next
            odd = odd.next

            even.next = even.next.next
            even = even.next

        odd.next = evenHead
        return head


def is_equal(l1, l2):
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1 = l1.next
        l2 = l2.next
    return l1 is None and l2 is None


if __name__ == "__main__":
    node5 = ListNode(5, None)
    node4 = ListNode(4, node5)
    node3 = ListNode(3, node4)
    node2 = ListNode(2, node3)
    node1 = ListNode(1, node2)
    head1 = node1

    node5 = ListNode(4, None)
    node4 = ListNode(2, node5)
    node3 = ListNode(5, node4)
    node2 = ListNode(3, node3)
    node1 = ListNode(1, node2)
    expexted_head1 = node1

    node7 = ListNode(7, None)
    node6 = ListNode(4, node7)
    node5 = ListNode(6, node6)
    node4 = ListNode(5, node5)
    node3 = ListNode(3, node4)
    node2 = ListNode(1, node3)
    node1 = ListNode(2, node2)
    head2 = node1

    node7 = ListNode(4, None)
    node6 = ListNode(5, node7)
    node5 = ListNode(1, node6)
    node4 = ListNode(7, node5)
    node3 = ListNode(6, node4)
    node2 = ListNode(3, node3)
    node1 = ListNode(2, node2)
    expexted_head2 = node1

    slt = Solution()
    assert is_equal(slt.oddEvenList(head1), expexted_head1)
    assert is_equal(slt.oddEvenList(head2), expexted_head2)

# Done ✅
