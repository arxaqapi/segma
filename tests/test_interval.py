from segma.structs.interval import Intervals


def test_Intervals_init():
    _int = Intervals()


def test_Intervals_add():
    intervals = Intervals()

    dummy_interval_list = [
        (0, 10, "a"),
        (0, 10, "b"),
        (20, 30, "a"),
        (20, 30, "c"),
        (25, 35, "a"),
        (49, 73, "c"),
        (40, 50, "c"),
    ]

    for interval in dummy_interval_list:
        intervals.add(interval)

    # Intervals([(0, 10, 'a'), (0, 10, 'b'), (20, 30, 'c'), (20, 35, 'a'), (40, 73, 'c')])
    assert len(intervals.intervals) == 5
