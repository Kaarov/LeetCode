sii = "abc"
tii = "ahbgdc"
# sii = "b"
# tii = "abc"


class Solution:
    def isSubsequence(self, s, t):
        count = 0
        if not s:
            return True
        for i in t:
            if i == s[count]:
                count += 1
            if count == len(s):
                return True
        return False


test = Solution()
print(test.isSubsequence(sii, tii))

# Done ✅
