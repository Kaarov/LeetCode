from typing import List


class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        ans = []
        numbers = [i for i in range(1, len(grid) ** 2 + 1)]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] in numbers:
                    numbers.remove(grid[i][j])
                else:
                    ans.append(grid[i][j])
        ans.append(numbers[-1])
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.findMissingAndRepeatedValues([[1, 3], [2, 2]]))  # [2, 4]
    print(slt.findMissingAndRepeatedValues([[9, 1, 7], [8, 9, 2], [3, 4, 6]]))  # [9, 5]

# Done ✅
