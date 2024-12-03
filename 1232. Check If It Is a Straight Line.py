from typing import List


class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        gradient = [coordinates[1][0] - coordinates[0][0], coordinates[1][1] - coordinates[0][1]]

        for i in range(2, len(coordinates)):
            gradient_2 = [coordinates[i][0] - coordinates[0][0], coordinates[i][1] - coordinates[0][1]]

            if (gradient[0] * gradient_2[1]) - (gradient[1] * gradient_2[0]) != 0:
                return False

        return True


slt = Solution()
print(slt.checkStraightLine([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]))
