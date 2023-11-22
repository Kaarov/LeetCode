from typing import List

arr = [1, 2, 2, 1, 1, 3]


class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        ans = []
        count = {}
        for i in arr:
            count[i] = count.get(i, 0) + 1

        for i in count.values():
            if i in ans:
                return False
            ans.append(i)
        return True


slt = Solution()
print(slt.uniqueOccurrences(arr))

# Done ✅
