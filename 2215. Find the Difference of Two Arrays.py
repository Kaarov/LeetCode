from typing import List

nums1 = [1, 2, 3]
nums2 = [2, 4, 6]


class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        ans = []
        n1, n2 = set(nums1), set(nums2)
        ans.append([i for i in n1 if i not in n2])
        ans.append([i for i in n2 if i not in n1])

        return ans


slt = Solution()
print(slt.findDifference(nums1, nums2))

# Done ✅
