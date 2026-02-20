class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1

        return -1


if __name__ == "__main__":
    slt = Solution()
    assert slt.search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert slt.search([-1, 0, 3, 5, 9, 12], 2) == -1

# Done ✅
