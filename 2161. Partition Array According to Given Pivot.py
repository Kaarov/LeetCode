from typing import List


class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        less = []
        more = []
        pivots = []
        for num in nums:
            if num < pivot:
                less.append(num)
            elif num == pivot:
                pivots.append(num)
            else:
                more.append(num)
        return less + pivots + more


if __name__ == '__main__':
    slt = Solution()
    print(slt.pivotArray(nums=[9, 12, 5, 10, 14, 3, 10], pivot=10))  # [9, 5, 3, 10, 10, 12, 14]
    print(slt.pivotArray(nums=[-3, 4, 3, 2], pivot=2))  # [-3, 2, 4, 3]

# Done ✅
