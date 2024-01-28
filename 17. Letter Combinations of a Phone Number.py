from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        ans = []
        digitToChar = {"2": "abc",
                       "3": "def",
                       "4": "ghi",
                       "5": "jkl",
                       "6": "mno",
                       "7": "qprs",
                       "8": "tuv",
                       "9": "wxyz"
                       }

        def backtrack(index, curStr):
            if len(curStr) == len(digits):
                ans.append(curStr)
                return
            for i in digitToChar[digits[index]]:
                backtrack(index + 1, curStr + i)

        if digits:
            backtrack(0, "")

        return ans


digits = "23"
slt = Solution()
print(slt.letterCombinations(digits))
