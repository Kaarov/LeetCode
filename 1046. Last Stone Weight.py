import heapq


class Solution:
    def lastStoneWeight(self, stones: list[int]) -> int:
        stones = [-stone for stone in stones]
        heapq.heapify(stones)

        while len(stones) > 1:
            highest_stone = heapq.heappop(stones)
            next_highest_stone = heapq.heappop(stones)

            if highest_stone != next_highest_stone:
                heapq.heappush(stones, highest_stone - next_highest_stone)

        stones.append(0)
        return abs(stones[0])


if __name__ == "__main__":
    slt = Solution()
    assert slt.lastStoneWeight([2, 7, 4, 1, 8, 1]) == 1
    assert slt.lastStoneWeight([1]) == 1

# Done ✅
