class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid

            if nums[l] <= nums[mid]:
                if target > nums[mid] or target < nums[l]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if target < nums[mid] or target > nums[r]:
                    r = mid - 1
                else:
                    l = mid + 1

        return -1


if __name__ == "__main__":
    slt = Solution()
    assert slt.search([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert slt.search([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert slt.search([1], 0) == -1
