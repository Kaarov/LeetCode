from typing import List


class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        ans = {}
        for idx, num in nums1:
            ans[idx] = ans.get(idx, 0) + num

        for idx, num in nums2:
            ans[idx] = ans.get(idx, 0) + num

        return sorted([[key, value] for key, value in ans.items()], key=lambda x: x[0])


if __name__ == '__main__':
    slt = Solution()
    print(slt.mergeArrays(nums1=[[1, 2], [2, 3], [4, 5]], nums2=[[1, 4], [3, 2], [4, 1]]))  # [[1, 6], [2, 3], [3, 2], [4, 6]]
    print(slt.mergeArrays(nums1=[[2, 4], [3, 6], [5, 5]], nums2=[[1, 3], [4, 3]]))  # [[1, 3], [2, 4], [3, 6], [4, 3], [5, 5]]

# Done ✅
