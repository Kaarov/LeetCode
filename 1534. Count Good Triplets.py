from typing import List


class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        ans: int = 0
        length: int = len(arr)
        for i in range(length - 2):
            for j in range(i + 1, length - 1):
                for k in range(j + 1, length):
                    if (abs(arr[i] - arr[j]) <= a) and (abs(arr[j] - arr[k]) <= b) and (abs(arr[i] - arr[k]) <= c):
                        ans += 1
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.countGoodTriplets(arr=[3, 0, 1, 1, 9, 7], a=7, b=2, c=3))  # 4
    print(slt.countGoodTriplets(arr=[1, 1, 2, 2, 3], a=0, b=0, c=1))  # 0

# Done ✅
