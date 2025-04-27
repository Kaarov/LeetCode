class Solution:
    def countValidWords(self, sentence: str) -> int:
        def checkValidWord(word: str) -> bool:
            if word[-1] in "!.,":
                word = word[:-1]

            hyphens = word.count("-")
            if hyphens:
                if hyphens > 1:
                    return False

                hyphen_idx = word.find("-")
                if hyphen_idx == 0 or hyphen_idx == len(word) - 1:
                    return False

                word = word.replace("-", "")

            if word.isalpha() and word == word.lower() or word == "":
                return True
            return False

        ans = 0
        for i in sentence.split():
            if checkValidWord(i):
                ans += 1
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.countValidWords("cat and  dog"))  # 3
    print(slt.countValidWords("!this  1-s b8d!"))  # 0
    print(slt.countValidWords("alice and  bob are playing stone-game10"))  # 5
    print(slt.countValidWords("he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."))  # 6

# Done ✅
