from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head

        while curr:
            while curr.next and curr.next.val == curr.val:
                curr.next = curr.next.next
            curr = curr.next

        return head


if __name__ == '__main__':
    node3 = ListNode(2, None)
    node2 = ListNode(1, node3)
    node1 = ListNode(1, node2)
    head1 = node1

    node5 = ListNode(3, None)
    node4 = ListNode(3, node5)
    node3 = ListNode(2, node4)
    node2 = ListNode(1, node3)
    node1 = ListNode(1, node2)
    head2 = node1

    slt = Solution()
    print(slt.deleteDuplicates(head1))
    print(slt.deleteDuplicates(head2))

# Done ✅
