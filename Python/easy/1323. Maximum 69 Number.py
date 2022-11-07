number = 9669


class Solution:
    def maximum69Number (self, num: int) -> int:
        num = str(num)
        ans = list(num)
        for i in range(len(ans)):
            if ans[i] == '6':
                ans[i] = '9'
                ans = int("".join(ans))
                return ans
        ans = int("".join(ans))
        return ans


slt = Solution()
print(slt.maximum69Number(number))
