class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])


s = "luffy is still joyboy"
slt = Solution()
print(slt.lengthOfLastWord(s))

# Done ✅
