class Solution:
    def minimumAbsDifference(self, arr: list[int]) -> list[list[int]]:
        arr.sort()
        min_diff = min(b - a for a, b in zip(arr, arr[1:]))
        return [[a, b] for a, b in zip(arr, arr[1:]) if b - a == min_diff]


if __name__ == "__main__":
    slt = Solution()
    assert slt.minimumAbsDifference([4, 2, 1, 3]) == [[1, 2], [2, 3], [3, 4]]
    assert slt.minimumAbsDifference([1, 3, 6, 10, 15]) == [[1, 3]]
    assert slt.minimumAbsDifference([3, 8, -10, 23, 19, -4, -14, 27]) == [[-14, -10], [19, 23], [23, 27]]
    assert slt.minimumAbsDifference([40, 11, 26, 27, -20]) == [[26, 27]]
