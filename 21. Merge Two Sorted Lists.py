from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = ans = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list2 or list2
        return ans.next


node3 = ListNode(4, None)
node2 = ListNode(2, node3)
node1 = ListNode(1, node2)
list1 = node1

node3 = ListNode(4, None)
node2 = ListNode(3, node3)
node1 = ListNode(1, node2)
list2 = node1

slt = Solution()
print(slt.mergeTwoLists(list1, list2))

# Done ✅
