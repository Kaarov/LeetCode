from functools import reduce
from typing import List


class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            curr = list(str(num))
            lrg = max(curr)
            ans += int(lrg * len(curr))
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.sumOfEncryptedInt([1, 2, 3]))  # 6
    print(slt.sumOfEncryptedInt([10, 21, 31]))  # 66

# Done ✅
