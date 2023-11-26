from collections import deque

senate = "RRDRDD"


class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        r_queue = deque()
        d_queue = deque()
        for i, c in enumerate(senate):
            if c == 'R':
                r_queue.append(i)
            else:
                d_queue.append(i)

        while r_queue and d_queue:
            dTurn = d_queue.popleft()
            rTurn = r_queue.popleft()
            if rTurn < dTurn:
                r_queue.append(dTurn + len(senate))
            else:
                d_queue.append(rTurn + len(senate))

        return 'Dire' if d_queue else 'Radiant'


slt = Solution()
print(slt.predictPartyVictory(senate))
