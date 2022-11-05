string1 = "ABCABC"
string2 = "ABC"


class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        ans = ""
        index = 65
        while True:
            check = ans + chr(index)
            if (check in str1) and (check in str2):
                ans += chr(index)
                index += 1
            else:
                return ans


slt = Solution()
print(slt.gcdOfStrings(string1, string2))
