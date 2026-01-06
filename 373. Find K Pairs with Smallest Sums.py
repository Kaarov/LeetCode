import heapq


class Solution:
    def kSmallestPairs(self, nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
        heap = []
        ans = []

        for i in range(len(nums1)):
            heapq.heappush(heap, (nums1[i] + nums2[0], nums1[i], nums2[0], 0))

        while heap and len(ans) < k:
            _, num1, num2, idx = heapq.heappop(heap)
            ans.append([num1, num2])

            if idx + 1 < len(nums2):
                heapq.heappush(heap, (num1 + nums2[idx + 1], num1, nums2[idx + 1], idx + 1))

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.kSmallestPairs([1, 7, 11], [2, 4, 6], 3) == [[1, 2], [1, 4], [1, 6]]
    assert slt.kSmallestPairs([1, 1, 2], [1, 2, 3], 2) == [[1, 1], [1, 1]]
