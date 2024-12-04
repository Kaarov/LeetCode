class Solution:
    def addBinary(self, a: str, b: str) -> str:
        ans = list(str(int(a) + int(b)))
        ans.insert(0, "0")
        for i in range(len(ans) - 1, 0, -1):
            if ans[i] == "3":
                ans[i] = "1"
                ans[i - 1] = str(int(ans[i - 1]) + 1)
            elif ans[i] == "2":
                ans[i] = "0"
                ans[i - 1] = str(int(ans[i - 1]) + 1)
        return str(int(''.join(ans)))


slt = Solution()
print(slt.addBinary(a='11', b='1'))

# Done ✅
