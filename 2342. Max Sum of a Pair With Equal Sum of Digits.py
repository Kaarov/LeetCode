from typing import List
from collections import defaultdict


class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        numbers = defaultdict(list)
        for i in range(len(nums)):
            number = 0
            for digit in str(nums[i]):
                number += int(digit)
            numbers[number].append(nums[i])
        ans = 0
        count = 0
        for number in numbers.values():
            if len(number) <= 1:
                count += 1
            else:
                max_num = max(number)
                number.remove(max_num)
                max_num += max(number)
                ans = max(ans, max_num)
        if count == len(numbers):
            return -1
        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.maximumSum([18, 43, 36, 13, 7]))  # 54
    print(slt.maximumSum([10, 12, 19, 14]))  # -1
    print(slt.maximumSum(
        [229, 398, 269, 317, 420, 464,
         491, 218, 439, 153, 482, 169,
         411, 93, 147, 50, 347, 210, 251,
         366, 401]
    ))  # 973

# Done ✅
