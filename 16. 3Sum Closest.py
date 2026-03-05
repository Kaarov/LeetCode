class Solution:
    def threeSumClosest(self, nums: list[int], target: int) -> int:
        nums.sort()
        closest = float("inf")

        for i in range(len(nums) - 2):
            if i != 0 and nums[i] == nums[i - 1]:
                continue

            l, r = i + 1, len(nums) - 1
            while l < r:
                three_sum = nums[i] + nums[l] + nums[r]
                if abs(three_sum - target) < abs(closest - target):
                    closest = three_sum
                if three_sum < target:
                    l += 1
                else:
                    r -= 1

        return closest


if __name__ == "__main__":
    slt = Solution()
    assert slt.threeSumClosest([-1, 2, 1, -4], 1) == 2
    assert slt.threeSumClosest([0, 0, 0], 1) == 0
