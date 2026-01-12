class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False

        return s in goal * 2


if __name__ == "__main__":
    slt = Solution()
    assert slt.rotateString(s="abcde", goal="cdeab") == True
    assert slt.rotateString(s="abcde", goal="abced") == False
    assert slt.rotateString(s="aa", goal="a") == False
    assert slt.rotateString(s="aa", goal="aaa") == False

# Done ✅
