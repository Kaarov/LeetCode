from typing import List

# chars = ["a", "a", "b", "b", "c", "c", "c"]
chars = ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]


class Solution:
    def compress(self, chars: List[str]) -> int:
        index = 0
        count = 1
        for i in range(1, len(chars) + 1):
            if i < len(chars) and chars[i - 1] == chars[i]:
                count += 1
            else:
                chars[index] = chars[i - 1]
                index += 1
                if count > 1:
                    for j in str(count):
                        chars[index] = j
                        index += 1
                count = 1
        return index


slt = Solution()
ans = slt.compress(chars)
print(ans)
print(chars[:ans])
