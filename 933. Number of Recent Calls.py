t = [[1], [100], [3001], [3002]]


class RecentCounter:

    def __init__(self):
        self.queue = []
        self.head = 0
        self.tail = 0

    def ping(self, t: int) -> int:
        self.queue.append(t)
        self.tail += 1
        while self.queue[self.head] < t - 3000:
            self.head += 1
        return self.tail - self.head


obj = RecentCounter()
for i in t:
    param_1 = obj.ping(i[0])
    print(param_1)

# Done ✅
