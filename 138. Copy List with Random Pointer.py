class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        oldNode = {None: None}

        cur = head
        while cur:
            newNode = Node(cur.val)
            oldNode[cur] = newNode
            cur = cur.next

        cur = head
        while cur:
            newNode = oldNode[cur]
            newNode.next = oldNode[cur.next]
            newNode.random = oldNode[cur.random]
            cur = cur.next

        return oldNode[head]


def is_deepcopy_equal(l1, l2):
    def get_nodes(head):
        nodes = []
        node_to_idx = {}
        cur = head
        idx = 0
        while cur:
            nodes.append(cur)
            node_to_idx[cur] = idx
            cur = cur.next
            idx += 1
        return nodes, node_to_idx

    nodes1, idx1 = get_nodes(l1)
    nodes2, idx2 = get_nodes(l2)

    if len(nodes1) != len(nodes2):
        return False

    for i in range(len(nodes1)):
        if nodes1[i].val != nodes2[i].val:
            return False

        if (i < len(nodes1) - 1 and nodes2[i].next != nodes2[i + 1]) or (
                i == len(nodes1) - 1 and nodes2[i].next is not None):
            return False

        r1 = nodes1[i].random
        r2 = nodes2[i].random
        if r1 is None and r2 is not None:
            return False
        if r1 is not None and r2 is None:
            return False
        if r1 is not None and r2 is not None:
            if idx1[r1] != idx2[r2]:
                return False

        if nodes1[i] is nodes2[i]:
            return False
    return True


if __name__ == "__main__":
    slt = Solution()

    n1 = Node(7)
    n2 = Node(13)
    n3 = Node(11)
    n4 = Node(10)
    n5 = Node(1)
    n1.next = n2; n2.next = n3; n3.next = n4; n4.next = n5
    n1.random = None
    n2.random = n1
    n3.random = n5
    n4.random = n3
    n5.random = n1

    copied_head = slt.copyRandomList(n1)
    assert is_deepcopy_equal(copied_head, slt.copyRandomList(n1))
