a = -2147483648
b = -1


class Solution:
    def divide(self, dnd: int, dvr: int) -> int:
        result = str(dnd/dvr)
        ans = ''
        if dnd == -2147483648 and dvr == -1:
            return 2147483647
        for x in result:
            if x != '.':
                ans += x
            else:
                break
        ans = int(ans)
        return ans


s = Solution()
print(s.divide(a, b))

# Done ✅
