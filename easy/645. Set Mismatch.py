numbers = [1, 2, 2, 4]


class Solution:
    def findErrorNums(self, nums):
        check = []
        for i in range(len(nums)):
            check.append(0)

        for i in nums:
            check[i - 1] += 1

        ans = [check.index(2) + 1, check.index(0) + 1]

        return ans


slt = Solution()
print(slt.findErrorNums(numbers))

# Done ✅
