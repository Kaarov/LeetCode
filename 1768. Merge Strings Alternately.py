word1 = "abcd"
word2 = "pq"


class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        min_len = min(len(word1), len(word2))
        max_val = word1 if max(len(word1), len(word2)) == len(word1) else word2
        ans = ""
        for i in range(min_len):
            ans += word1[i] + word2[i]

        ans += max_val[min_len:]

        return ans


slt = Solution()
print(slt.mergeAlternately(word1, word2))

# Done ✅
