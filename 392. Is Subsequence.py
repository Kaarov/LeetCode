s = "abc"
t = "ahbgdc"


class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        for i in t:
            if i == s[0]:
                s = s[1:]
            if not s:
                return True
        return False


slt = Solution()
print(slt.isSubsequence(s, t))

# Done ✅
