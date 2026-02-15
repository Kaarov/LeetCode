from heapq import heappush, heappop


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def is_equal(l1: ListNode, l2: ListNode) -> bool:
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1 = l1.next
        l2 = l2.next
    return l1 is None and l2 is None


class Solution:
    def insertionSortList(self, head: ListNode | None) -> ListNode | None:
        heap = []
        while (head):
            heappush(heap, head.val)
            head = head.next
        sorted = ListNode(0, None)
        dummy = sorted
        while (heap):
            sorted.val = heappop(heap)
            if (heap):
                sorted.next = ListNode(None)
            sorted = sorted.next
        return dummy


if __name__ == "__main__":
    slt = Solution()

    node3 = ListNode(3, None)
    node2 = ListNode(1, node3)
    node1 = ListNode(2, node2)
    head1 = ListNode(4, node1)

    node3 = ListNode(4, None)
    node2 = ListNode(3, node3)
    node1 = ListNode(2, node2)
    expected_head1 = ListNode(1, node1)

    node4 = ListNode(0, None)
    node3 = ListNode(4, node4)
    node2 = ListNode(3, node3)
    node1 = ListNode(5, node2)
    head2 = ListNode(-1, node1)

    node4 = ListNode(5, None)
    node3 = ListNode(4, node4)
    node2 = ListNode(3, node3)
    node1 = ListNode(0, node2)
    expected_head2 = ListNode(-1, node1)

    assert is_equal(slt.insertionSortList(head1), expected_head1)
    assert is_equal(slt.insertionSortList(head2), expected_head2)
