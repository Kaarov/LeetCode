from typing import List

spells = [5, 1, 3]
potions = [1, 2, 3, 4, 5]
success = 7


class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()
        ans = []
        for i in spells:
            l, r = 0, len(potions) - 1
            count = len(potions)
            while l <= r:
                m = (l + r) // 2
                if potions[m] * i >= success:
                    r = m - 1
                    count = m
                else:
                    l = m + 1
            ans.append((len(potions)) - count)
        return ans


slt = Solution()
print(slt.successfulPairs(spells, potions, success))
