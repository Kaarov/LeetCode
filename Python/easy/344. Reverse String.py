string = ["h", "e", "l", "l", "o"]


class Solution:
    def reverseString(self, s: List[str]) -> None:
        s.reverse()
        return s


slt = Solution()
print(slt.reverseString(string))
