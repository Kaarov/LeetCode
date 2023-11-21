from typing import List

gain = [-5, 1, 5, 0, -7]


class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        ans = [0, ]
        for i in range(len(gain)):
            ans += [ans[i] + gain[i]]
        return max(ans)


slt = Solution()
print(slt.largestAltitude(gain))

# Done ✅
