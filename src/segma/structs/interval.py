from collections import defaultdict
from typing import Iterator, TypeAlias

# TODO - fix type hints (support floats)
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
        for s, e, label in intervals[1:]:
            # Check if current interval overlaps or is adjacent to the last one
            if s <= ret[-1][1]:  # Changed from > to <= to merge adjacent intervals
                # Merge by extending the end of the last interval
                ret[-1] = (ret[-1][0], max(ret[-1][1], e), label)
            else:
                # No overlap, add as new interval
                ret.append((s, e, label))
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

    def __iter__(self) -> Iterator[Interval]:
        return iter(self.intervals)

    def __len__(self):
        return len(self.intervals)
