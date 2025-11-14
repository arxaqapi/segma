from segma.structs.interval import Intervals


def test_empty_intervals():
    """Test empty interval collection."""
    intervals = Intervals()
    assert intervals.intervals == []


def test_single_interval():
    """Test adding a single interval."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    assert intervals.intervals == [(0, 10, "a")]


def test_adjacent_intervals_same_label():
    """Test merging adjacent intervals with the same label."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((10, 20, "a"))
    assert intervals.intervals == [(0, 20, "a")]

    # Test with multiple adjacent intervals
    intervals = Intervals()
    intervals.add((0, 5, "b"))
    intervals.add((5, 10, "b"))
    intervals.add((10, 15, "b"))
    assert intervals.intervals == [(0, 15, "b")]


def test_overlapping_intervals_same_label():
    """Test merging overlapping intervals with the same label."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((5, 15, "a"))
    assert intervals.intervals == [(0, 15, "a")]

    # Complete overlap
    intervals = Intervals()
    intervals.add((0, 20, "a"))
    intervals.add((5, 10, "a"))
    assert intervals.intervals == [(0, 20, "a")]

    # Partial overlap
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((8, 20, "a"))
    assert intervals.intervals == [(0, 20, "a")]


def test_non_overlapping_intervals_same_label():
    """Test non-overlapping, non-adjacent intervals with same label."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((15, 25, "a"))
    assert intervals.intervals == [(0, 10, "a"), (15, 25, "a")]

    intervals = Intervals()
    intervals.add((0, 5, "a"))
    intervals.add((10, 15, "a"))
    intervals.add((20, 25, "a"))
    assert intervals.intervals == [(0, 5, "a"), (10, 15, "a"), (20, 25, "a")]


def test_intervals_different_labels():
    """Test intervals with different labels should not merge."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((10, 20, "b"))
    assert intervals.intervals == [(0, 10, "a"), (10, 20, "b")]

    # Overlapping with different labels
    intervals = Intervals()
    intervals.add((0, 15, "a"))
    intervals.add((10, 20, "b"))
    assert intervals.intervals == [(0, 15, "a"), (10, 20, "b")]


def test_mixed_labels_complex():
    """Test complex scenarios with multiple labels."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((5, 15, "b"))
    intervals.add((10, 20, "a"))
    intervals.add((12, 18, "b"))

    # Should merge intervals with same labels
    expected = [(0, 20, "a"), (5, 18, "b")]
    assert intervals.intervals == expected


def test_numeric_labels():
    """Test intervals with numeric labels."""
    intervals = Intervals()
    intervals.add((0, 10, 1))
    intervals.add((10, 20, 1))
    assert intervals.intervals == [(0, 20, 1)]

    intervals = Intervals()
    intervals.add((0, 10, 1))
    intervals.add((5, 15, 2))
    intervals.add((10, 20, 1))
    expected = [(0, 20, 1), (5, 15, 2)]
    assert intervals.intervals == expected


def test_mixed_numeric_string_labels():
    """Test intervals with mixed numeric and string labels."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((5, 15, 1))
    intervals.add((10, 20, "a"))
    intervals.add((15, 25, 1))

    expected = [(0, 20, "a"), (5, 25, 1)]
    assert intervals.intervals == expected


def test_single_point_intervals():
    """Test intervals with same start and end (single points)."""
    intervals = Intervals()
    intervals.add((5, 5, "a"))
    intervals.add((5, 5, "a"))
    assert intervals.intervals == [(5, 5, "a")]

    intervals = Intervals()
    intervals.add((5, 5, "a"))
    intervals.add((5, 10, "a"))
    assert intervals.intervals == [(5, 10, "a")]


def test_unordered_insertion():
    """Test adding intervals in non-sorted order."""
    intervals = Intervals()
    intervals.add((20, 30, "a"))
    intervals.add((0, 10, "a"))
    intervals.add((10, 20, "a"))
    assert intervals.intervals == [(0, 30, "a")]

    intervals = Intervals()
    intervals.add((15, 20, "b"))
    intervals.add((5, 10, "a"))
    intervals.add((0, 5, "a"))
    intervals.add((10, 15, "b"))
    expected = [(0, 10, "a"), (10, 20, "b")]
    assert intervals.intervals == expected


def test_large_chain_merge():
    """Test merging a long chain of adjacent/overlapping intervals."""
    intervals = Intervals()
    for i in range(0, 50, 5):
        intervals.add((i, i + 10, "a"))
    # All intervals should merge into one
    assert intervals.intervals == [(0, 55, "a")]


def test_negative_coordinates():
    """Test intervals with negative coordinates."""
    intervals = Intervals()
    intervals.add((-10, 0, "a"))
    intervals.add((0, 10, "a"))
    assert intervals.intervals == [(-10, 10, "a")]

    intervals = Intervals()
    intervals.add((-20, -10, "a"))
    intervals.add((-15, -5, "a"))
    assert intervals.intervals == [(-20, -5, "a")]


def test_very_large_coordinates():
    """Test intervals with very large coordinates."""
    intervals = Intervals()
    intervals.add((0, 1000000, "a"))
    intervals.add((1000000, 2000000, "a"))
    assert intervals.intervals == [(0, 2000000, "a")]


def test_multiple_add_operations():
    """Test multiple sequential add operations."""
    intervals = Intervals()

    # Add first interval
    intervals.add((0, 5, "a"))
    assert intervals.intervals == [(0, 5, "a")]

    # Add non-adjacent interval
    intervals.add((10, 15, "a"))
    assert intervals.intervals == [(0, 5, "a"), (10, 15, "a")]

    # Bridge the gap
    intervals.add((5, 10, "a"))
    assert intervals.intervals == [(0, 15, "a")]

    # Add different label
    intervals.add((7, 12, "b"))
    assert intervals.intervals == [(0, 15, "a"), (7, 12, "b")]


def test_subset_intervals():
    """Test intervals that are complete subsets of others."""
    intervals = Intervals()
    intervals.add((0, 20, "a"))
    intervals.add((5, 10, "a"))
    intervals.add((12, 15, "a"))
    assert intervals.intervals == [(0, 20, "a")]


def test_identical_intervals():
    """Test adding identical intervals."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((0, 10, "a"))
    intervals.add((0, 10, "a"))
    assert intervals.intervals == [(0, 10, "a")]


def test_edge_case_one_unit_gap():
    """Test intervals with exactly one unit gap (should not merge)."""
    intervals = Intervals()
    intervals.add((0, 10, "a"))
    intervals.add((11, 20, "a"))
    assert intervals.intervals == [(0, 10, "a"), (11, 20, "a")]


def test_special_characters_in_labels():
    """Test intervals with special characters in string labels."""
    intervals = Intervals()
    intervals.add((0, 10, "a-b"))
    intervals.add((10, 20, "a-b"))
    assert intervals.intervals == [(0, 20, "a-b")]

    intervals = Intervals()
    intervals.add((0, 10, "label with spaces"))
    intervals.add((5, 15, "label with spaces"))
    assert intervals.intervals == [(0, 15, "label with spaces")]


def test_empty_string_label():
    """Test intervals with empty string as label."""
    intervals = Intervals()
    intervals.add((0, 10, ""))
    intervals.add((10, 20, ""))
    assert intervals.intervals == [(0, 20, "")]
