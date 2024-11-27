from typing import List


class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        ans = list(map(sum, accounts))
        return max(ans)


slt = Solution()
print(slt.maximumWealth(accounts=[[1, 5], [7, 3], [3, 5]]))

# Done ✅
