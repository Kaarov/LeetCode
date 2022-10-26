num = [2, 2, 1]


class Solution:
    def singleNumber(self, nums):
        ans = []
        a = 0
        count = 0
        while True:
            for y in nums:
                if nums[count] == y:
                    a += 1
            if a == 1:
                ans.append(nums[count])
            a = 0
            count += 1
            if count == len(nums):
                break
        return ans[0]


print(Solution().singleNumber(num))

# Not Done Yet ❌
