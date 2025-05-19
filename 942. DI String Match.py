from typing import List


class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        ans = []
        low = 0
        high = len(s)
        for i in s:
            if i == 'I':
                ans.append(low)
                low += 1
            else:
                ans.append(high)
                high -= 1
        return ans + [low]


if __name__ == '__main__':
    slt = Solution()
    print(slt.diStringMatch("IDID"))  # [0, 4, 1, 3, 2]
    print(slt.diStringMatch("III"))  # [0, 1, 2, 3]
    print(slt.diStringMatch("DDI"))  # [3, 2, 0, 1]

# Done ✅
