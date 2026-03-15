class Solution:
    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        ans = set()
        l = 0

        for r in range(len(nums)):
            if r - l > k:
                ans.remove(nums[l])
                l += 1
            if nums[r] in ans:
                return True
            ans.add(nums[r])

        return False


if __name__ == "__main__":
    slt = Solution()
    assert slt.containsNearbyDuplicate([1, 2, 3, 1], 3) == True
    assert slt.containsNearbyDuplicate([1, 0, 1, 1], 1) == True
    assert slt.containsNearbyDuplicate([1, 2, 3, 1, 2, 3], 2) == False
