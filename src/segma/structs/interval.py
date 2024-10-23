from collections import defaultdict
from typing import TypeAlias

Interval: TypeAlias = tuple[int, int, int | str]


class Intervals:
    """Adapted from interlap.Interval to support multiple labels."""

    def __init__(self) -> None:
        # list of tuple
        self.intervals: list[Interval] = []

    def add(self, interval: Interval):
        """adds an interval to the inner list of intervals, extends the"""
        self.intervals = self._reduce_per_label(self.intervals + [interval])

    def _reduce(self, intervals: list[Interval]) -> list[Interval]:
        if len(intervals) < 2:
            return intervals

        intervals.sort()

        ret = [intervals[0]]
        for next_i, (s, e, label) in enumerate(intervals, start=1):
            if next_i == len(intervals):
                ret[-1] = ret[-1][0], max(ret[-1][1], e), label
                break

            ns, ne, label = intervals[next_i]
            if e > ns or ret[-1][1] > ns:
                ret[-1] = ret[-1][0], max(e, ne, ret[-1][1]), label
            else:
                ret.append((ns, ne, label))
        return ret

    def _reduce_per_label(self, intervals: list[Interval]) -> list[Interval]:
        """merge intervals if they have the same label associated with them"""
        map: dict[str | int, list[Interval]] = defaultdict(list)
        for start, end, label in intervals:
            map[label].append((start, end, label))

        final: list[list[Interval]] = [
            self._reduce(sub_intervals) for _, sub_intervals in map.items()
        ]
        return sorted(sum(final, []))

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    def __len__(self):
        return len(self.intervals)
