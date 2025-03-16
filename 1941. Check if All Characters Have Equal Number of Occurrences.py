from collections import Counter


class Solution:
    def areOccurrencesEqual(self, s: str) -> bool:
        c = Counter(s)
        ans = c[s[0]]
        for i in c.values():
            if i != ans:
                return False
        return True


if __name__ == '__main__':
    slt = Solution()
    print(slt.areOccurrencesEqual("abacbc"))  # True
    print(slt.areOccurrencesEqual("aaabb"))  # False

# Done ✅
