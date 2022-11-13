string = " the sky is blue "


class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.split()
        s = s[::-1]
        s = " ".join(s)
        return s


slt = Solution()
print(slt.reverseWords(string))

# Done ✅
