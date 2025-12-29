class MyQueue:
    def __init__(self):
        self.queue = []

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        return self.queue.pop(0)

    def peek(self) -> int | None:
        if len(self.queue) > 0:
            return self.queue[0]
        return None

    def empty(self) -> bool:
        return len(self.queue) == 0


if __name__ == '__main__':
    # Your MyQueue object will be instantiated and called as such:
    obj = MyQueue()
    obj.push(1)
    obj.push(2)
    param_1 = obj.pop()
    param_2 = obj.peek()
    param_3 = obj.empty()
    param_4 = obj.pop()
    param_5 = obj.peek()
    param_6 = obj.empty()

    assert param_1 == 1
    assert param_2 == 2
    assert param_3 == False
    assert param_4 == 2
    assert param_5 == None
    assert param_6 == True

# Done ✅
