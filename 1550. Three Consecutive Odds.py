from typing import List


class Solution:
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        for i in range(len(arr) - 2):
            if arr[i] % 2 and arr[i + 1] % 2 and arr[i + 2] % 2:
                return True
        return False


if __name__ == '__main__':
    slt = Solution()
    print(slt.threeConsecutiveOdds([2, 6, 4, 1]))  # False
    print(slt.threeConsecutiveOdds([1, 2, 34, 3, 4, 5, 7, 23, 12]))  # True
    print(slt.threeConsecutiveOdds([1, 2, 3]))  # False
    print(slt.threeConsecutiveOdds([1, 1, 1]))  # True
    print(slt.threeConsecutiveOdds([102, 780, 919, 897, 901]))  # True
