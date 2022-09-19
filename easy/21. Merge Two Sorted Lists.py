list1 = [1, 2, 4]
list2 = [1, 3, 4]


# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, l1, l2):
        dummy = temp = ListNode(0)
        while l1 != None and l2 != None:

            if l1.val < l2.val:
                temp.next = l1
                l1 = l1.next
            else:
                temp.next = l2
                l2 = l2.next
            temp = temp.next
        temp.next = l1 or l2
        return dummy.next


node3 = ListNode(4, None)
node2 = ListNode(2, node3)
node1 = ListNode(1, node2)
list1 = node1

node3 = ListNode(4, None)
node2 = ListNode(3, node3)
node1 = ListNode(1, node2)
list2 = node1

a = Solution()
print(a.mergeTwoLists(list1, list2))
