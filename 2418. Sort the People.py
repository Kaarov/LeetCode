from typing import List


class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        ans = list(zip(heights, names))
        ans.sort(reverse=True)
        return [name for _, name in ans]


if __name__ == '__main__':
    slt = Solution()
    print(slt.sortPeople(
        names=["Mary", "John", "Emma"],
        heights=[180, 165, 170]
    ))  # ["Mary", "Emma", "John"]
    print(slt.sortPeople(
        names=["Alice", "Bob", "Bob"],
        heights=[155, 185, 150]
    ))  # ["Bob", "Alice", "Bob"]

# Done ✅
