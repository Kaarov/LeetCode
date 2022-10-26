# numbers = [1, 2, 3, 1]
# kk = 3
numbers = [1, 2, 3, 1, 2, 3]
kk = 2


class Solution:
    def containsNearbyDuplicate(self, nums, k):
        ans = dict()
        for i in range(len(nums)):
            if nums[i] not in ans:
                ans[nums[i]] = [i]
            else:
                if abs(ans[nums[i]][-1] - i) <= k:
                    return True
                else:
                    ans[nums[i]].append(i)
        return False


slt = Solution()
print(slt.containsNearbyDuplicate(numbers, kk))

# Done ✅
