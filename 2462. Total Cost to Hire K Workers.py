import heapq
from collections import deque
from typing import List


class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        if len(costs) >= 2 * candidates:
            L = costs[:candidates] + [float('inf')]
            heapq.heapify(L)
            R = costs[-candidates:] + [float('inf')]
            heapq.heapify(R)
            q = deque(costs[candidates:-candidates])
        else:
            return sum(heapq.nsmallest(k, costs))

        ans = 0
        for _ in range(k):
            if q:
                if L[0] <= R[0]:
                    ans += heapq.heappop(L)
                    heapq.heappush(L, q.popleft())
                else:
                    ans += heapq.heappop(R)
                    heapq.heappush(R, q.pop())
            else:
                if L[0] <= R[0]:
                    ans += heapq.heappop(L)
                else:
                    ans += heapq.heappop(R)
        return ans

