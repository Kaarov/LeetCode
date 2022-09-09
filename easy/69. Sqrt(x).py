class Solution:
    def mySqrt(self, x: int) -> int:
        result = str(x ** 0.5)
        ans = ''
        for x in result:
            if x != '.':
                ans += x
            else:
                break
        return int(ans)


a = int(input())
print(Solution().mySqrt(a))

# Done✅
