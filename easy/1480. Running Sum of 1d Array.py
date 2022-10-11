# numbers = [1, 2, 3, 4]
numbers = [1, 1, 1, 1, 1]


class Solution:
    def runningSum(self, nums):
        ans = []
        for i in range(1, len(nums) + 1):
            summa = 0
            for j in range(i):
                summa += nums[j]
            ans.append(summa)
        return ans


test = Solution()
print(test.runningSum(numbers))

# Done ✅
