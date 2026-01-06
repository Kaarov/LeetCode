import heapq


class Solution:
    def isPossible(self, target: list[int]) -> bool:
        if len(target) == 1: return target[0] == 1

        total = sum(target)
        heap = [-x for x in target]
        heapq.heapify(heap)

        while -heap[0] > 1:
            highest = -heapq.heappop(heap)
            rest = total - highest

            if rest == 1: return True
            if rest == 0 or highest <= rest: return False

            prev_val = highest % rest

            if prev_val == 0: return False

            total = rest + prev_val
            heapq.heappush(heap, -prev_val)
        return True


if __name__ == "__main__":
    slt = Solution()
    assert slt.isPossible([9, 3, 5]) == True
    assert slt.isPossible(target=[1, 1, 1, 2]) == False
    assert slt.isPossible([8, 5]) == True
