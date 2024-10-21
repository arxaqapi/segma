from segma.structs.interval import Intervals


def test_Intervals_init():
    _int = Intervals()


def test_Intervals_add():
    intervals = Intervals()

    l = [
        (0, 10, "a"),
        (0, 10, "b"),
        (20, 30, "a"),
        (20, 30, "c"),
        (25, 35, "a"),
        (49, 73, "c"),
        (40, 50, "c"),
    ]

    for interval in l:
        intervals.add(interval)

    # Intervals([(0, 10, 'a'), (0, 10, 'b'), (20, 30, 'c'), (20, 35, 'a'), (40, 73, 'c')])
    assert len(intervals.intervals) == 5
