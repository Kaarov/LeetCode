from typing import List


class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        morse_code = [
            ".-", "-...", "-.-.", "-..", ".", "..-.", "--.",
            "....", "..", ".---", "-.-", ".-..", "--", "-.",
            "---", ".--.", "--.-", ".-.", "...", "-", "..-",
            "...-", ".--", "-..-", "-.--", "--..",
        ]
        ans = []
        for word in words:
            morse = ""
            for char in word:
                morse += morse_code[ord(char) - 97]
            if morse not in ans:
                ans.append(morse)
        return len(ans)


if __name__ == '__main__':
    slt = Solution()
    print(slt.uniqueMorseRepresentations(["gin", "zen", "gig", "msg"]))  # 2
    print(slt.uniqueMorseRepresentations(["a"]))  # 1

# Done ✅
