from typing import List


class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort()

        merged_meetings = [meetings[0], ]
        for meeting in meetings[1:]:
            if meeting[0] > merged_meetings[-1][1]:
                merged_meetings.append(meeting)
            else:
                merged_meetings[-1][1] = max(merged_meetings[-1][1], meeting[1])

        meeting_days_count = 0
        for start, end in merged_meetings:
            meeting_days_count += end - start + 1

        return days - meeting_days_count


if __name__ == '__main__':
    slt = Solution()
    print(slt.countDays(days=10, meetings=[[5, 7], [1, 3], [9, 10]]))  # 2
    print(slt.countDays(days=5, meetings=[[2, 4], [1, 3]]))  # 1
    print(slt.countDays(days=6, meetings=[[1, 6]]))  # 0
