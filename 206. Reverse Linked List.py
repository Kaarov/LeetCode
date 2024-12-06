from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head

        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next

        return prev


node4 = ListNode(5, None)
node3 = ListNode(4, node4)
node2 = ListNode(3, node3)
node1 = ListNode(2, node2)
head = ListNode(1, node1)

slt = Solution()
print(slt.reverseList(head))

# Done ✅
