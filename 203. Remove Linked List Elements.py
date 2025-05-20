from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        while head and head.val == val:
            head = head.next
        curr = head

        while curr:
            while curr.next and curr.next.val == val:
                curr.next = curr.next.next
            curr = curr.next

        return head


if __name__ == "__main__":
    slt = Solution()
    node = ListNode(1,
                    ListNode(2,
                             ListNode(6,
                                      ListNode(3,
                                               ListNode(4,
                                                        ListNode(5,
                                                                 ListNode(6)
                                                                 )
                                                        )
                                               )
                                      )
                             )
                    )
    print(slt.removeElements(node, 6))  # [1, 2, 3, 4, 5]

    node = ListNode(7,
                    ListNode(7,
                             ListNode(7,
                                      ListNode(7)
                                      )
                             )
                    )
    print(slt.removeElements(node, 7))  # []

# Done ✅
