from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        list1 = ""
        list2 = ""

        while l1:
            list1 += str(l1.val)
            l1 = l1.next

        while l2:
            list2 += str(l2.val)
            l2 = l2.next

        cur = ans = ListNode()

        for i in str(int(list1) + int(list2)):
            cur.next = ListNode(int(i))
            cur = cur.next

        return ans.next


if __name__ == '__main__':
    node3 = ListNode(3, None)
    node2 = ListNode(4, node3)
    node1 = ListNode(2, node2)
    l1 = ListNode(7, node1)

    node2 = ListNode(4, None)
    node1 = ListNode(6, node2)
    l2 = ListNode(5, node1)

    slt = Solution()
    print(slt.addTwoNumbers(l1, l2))

# Done ✅
