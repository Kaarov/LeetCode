class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        nums = sorted(intervals, key=lambda n: n[0])
        i = 0

        while i < len(nums) - 1:
            if nums[i][1] >= nums[i + 1][0]:
                if nums[i][1] < nums[i + 1][1]:
                    nums[i][1] = nums[i + 1][1]
                nums.pop(i + 1)
            else:
                i += 1

        return nums


if __name__ == "__main__":
    slt = Solution()
    assert slt.merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert slt.merge([[1, 4], [4, 5]]) == [[1, 5]]
    assert slt.merge([[4, 7], [1, 4]]) == [[1, 7]]
    assert slt.merge([[1, 4], [2, 3]]) == [[1, 4]]

# Done ✅
# Note: Could be improved further
