class Solution:
    def peakIndexInMountainArray(self, arr: list[int]) -> int:
        peak = max(arr)

        for i, a in enumerate(arr):
            if a == peak:
                return i


if __name__ == "__main__":
    slt = Solution()
    assert slt.peakIndexInMountainArray([0, 1, 0]) == 1
    assert slt.peakIndexInMountainArray([0, 2, 1, 0]) == 1
    assert slt.peakIndexInMountainArray([0, 10, 5, 2]) == 1

# Done ✅
# Note: Could be improved further
