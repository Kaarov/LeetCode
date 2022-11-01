# number = 240
# key = 2
number = 430043
key = 2


class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        ans = 0
        range_i = len(str(num)) - k + 1
        for i in range(range_i):
            index = int(str(num)[i:i + k])
            if index != 0 and num % index == 0:
                ans += 1
        return ans


slt = Solution()
print(slt.divisorSubstrings(number, key))

# Done ✅
