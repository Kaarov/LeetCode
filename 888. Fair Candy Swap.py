from typing import List


class Solution:
    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        sum_alice = sum(aliceSizes)
        sum_bob = sum(bobSizes)

        delta = (sum_alice - sum_bob) // 2
        alice_set = set(aliceSizes)

        for i in bobSizes:
            if i + delta in alice_set:
                return [i + delta, i]


if __name__ == '__main__':
    slt = Solution()
    print(slt.fairCandySwap(aliceSizes=[1, 1], bobSizes=[2, 2]))  # [1, 2]
    print(slt.fairCandySwap(aliceSizes=[1, 2], bobSizes=[2, 3]))  # [1, 2]
    print(slt.fairCandySwap(aliceSizes=[2], bobSizes=[1, 3]))  # [2, 3]
