class SmallestInfiniteSet:

    def __init__(self):
        self.infinite_set = []
        self.count = 0

    def popSmallest(self) -> int:
        self.count += 1
        if self.count not in self.infinite_set:
            self.infinite_set.append(self.count)
        self.infinite_set.sort()
        return self.infinite_set.pop(0)

    def addBack(self, num: int) -> None:
        if num not in self.infinite_set:
            self.infinite_set.insert(0, num)

# Your SmallestInfiniteSet object will be instantiated and called as such:
# obj = SmallestInfiniteSet()
# param_1 = obj.popSmallest()
# obj.addBack(num)

# Done ✅
