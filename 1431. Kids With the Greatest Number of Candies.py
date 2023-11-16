from typing import List

candies = [2, 3, 5, 1, 3]
extraCandies = 3


class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_el = max(candies)
        for i in range(len(candies)):
            candies[i] = True if candies[i] + extraCandies >= max_el else False
        return candies


slt = Solution()
print(slt.kidsWithCandies(candies, extraCandies))
