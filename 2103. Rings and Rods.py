from collections import defaultdict


class Solution:
    def countPoints(self, rings: str) -> int:
        ans = 0
        points = defaultdict(set)

        for i in range(0, len(rings) // 2):
            color, rod = rings[i * 2], rings[i * 2 + 1]
            points[rod].add(color)

        for i in points.values():
            if len(i) == 3:
                ans += 1

        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.countPoints('B0B6G0R6R0R6G9'))

# Done ✅
