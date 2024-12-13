class Solution:
    def countVowelSubstrings(self, word: str) -> int:
        ans = 0
        vowels = {"a", "e", "i", "o", "u"}

        def checkVowelSubstring(word: str) -> bool:
            nonlocal ans
            for i in range(len(word) - 4):
                for j in range(len(word) - 4):
                    if set(word[i:j + 5]) == vowels:
                        ans += 1

        words = []
        result = ""
        for i in word:
            if i in vowels:
                result += i
            else:
                if len(result) > 5:
                    words.append(result)
                result = ""
        if len(result) >= 5:
            words.append(result)
        for word in words:
            checkVowelSubstring(word)
        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.countVowelSubstrings("aeiouu"))

# Done ✅
