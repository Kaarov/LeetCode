from typing import List


class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        if len(words) != len(s):
            return False
        for i in range(len(words)):
            if words[i][0] != s[i]:
                return False
        return True


if __name__ == '__main__':
    slt = Solution()
    print(slt.isAcronym(words=["alice", "bob", "charlie"], s="abc"))  # True
    print(slt.isAcronym(words=["an", "apple"], s="a"))  # False
    print(slt.isAcronym(words=["never", "gonna", "give", "up", "on", "you"], s="ngguoy"))  # True

# Done ✅
