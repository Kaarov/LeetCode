int1 = 30
int2 = 6


class Solution:
    def commonFactors(self, a: int, b: int):
        count = 0
        ans = []
        for i in range(1, max(a, b)+1):
            if min(a, b)+1 < i:
                break
            if a % i == 0 and b % i == 0:
                count += 1
                ans.append(i)
        return (count, ans)


test = Solution()
print(test.commonFactors(int1, int2))

# Done ✅
