from typing import List


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        ans = []
        for n1 in nums1:
            i = 0
            flag = True
            while i < len(nums2):
                if n1 == nums2[i]:
                    while i < len(nums2):
                        if n1 < nums2[i]:
                            ans.append(nums2[i])
                            flag = False
                            break
                        i += 1
                i += 1
            if flag: ans.append(-1)
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.nextGreaterElement(nums1=[4, 1, 2], nums2=[1, 3, 4, 2]))  # [-1, 3, -1]
    print(slt.nextGreaterElement(nums1=[2, 4], nums2=[1, 2, 3, 4]))  # [3, -1]

# Done ✅
