numbers = [2, 7, 11, 15]
tar = 9


class Solution:
    def twoSum(self, nums, target):
        ans = []
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    ans.append(i)
                    ans.append(j)
        return ans


slt = Solution()
print(slt.twoSum(numbers, tar))

# Done ✅
