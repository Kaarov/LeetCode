string = "Test1ng-Leet=code-Q!"


class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        check = ""
        ans = ""
        for i in range(len(s) - 1, -1, -1):
            if s[i].isalpha():
                check += s[i]
        count = 0
        index = 0
        while len(ans) != len(s):
            if not s[count].isalpha():
                ans += s[count]
                count += 1
            else:
                ans += check[index]
                index += 1
                count += 1
        return ans


slt = Solution()
print(slt.reverseOnlyLetters(string))

# Done ✅
