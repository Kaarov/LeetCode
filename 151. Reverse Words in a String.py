string = "  hello world  "


class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.split()[::-1]
        return " ".join(s)


slt = Solution()
print(slt.reverseWords(string))
