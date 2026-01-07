class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if word == word.capitalize(): return True
        capital = 0
        not_capital = 0

        for w in word:
            if 97 <= ord(w) <= 122:
                not_capital += 1
            else:
                capital += 1
            if capital > 0 and not_capital > 0: return False

        return True


if __name__ == "__main__":
    slt = Solution()
    assert slt.detectCapitalUse("USA") == True
    assert slt.detectCapitalUse("FlaG") == False
    assert slt.detectCapitalUse("FFFFFFFFFFFFFFFFFFFFf") == False

# Done ✅
# Note: Could be improved further
