word1 = "aaabbbbccddeeeeefffff"
word2 = "aaaaabbcccdddeeeeffff"


class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        w1 = {}
        w2 = {}

        for i in word1:
            w1[i] = w1.get(i, 0) + 1

        for i in word2:
            w2[i] = w2.get(i, 0) + 1

        return sorted(w1.values()) == sorted(w2.values()) and w1.keys() == set(w2.keys())


slt = Solution()
print(slt.closeStrings(word1, word2))

# Done ✅
