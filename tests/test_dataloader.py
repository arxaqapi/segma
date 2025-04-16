from pathlib import Path

from segma.data.loaders import filter_annotations, load_annotations


def test_load_annotations():
    aa_p = Path("tests/sample/test.aa")

    annots = load_annotations(aa_p)

    assert len(annots) == 6


def test_filter_annotations_unmodified():
    aa_p = Path("tests/sample/test.aa")

    annots = load_annotations(aa_p)
    unmodified = filter_annotations(
        annots, covered_labels=("child", "other_child", "female", "male")
    )
    assert unmodified == annots
    assert len(unmodified) == len(annots)


def test_filter_annotations():
    aa_p = Path("tests/sample/test.aa")

    annots = load_annotations(aa_p)

    no_child = filter_annotations(
        annots, covered_labels=("other_child", "female", "male")
    )
    assert len(no_child) == 4
    for annot in no_child:
        assert annot.label != "child"

    no_male = filter_annotations(
        annots, covered_labels=("child", "other_child", "female")
    )
    assert len(no_male) == 5
    for annot in no_male:
        assert annot.label != "male"

    # test chaining
    no_male_from_no_child = filter_annotations(
        no_child, covered_labels=("other_child", "female")
    )
    assert no_male_from_no_child == filter_annotations(
        annots, covered_labels=("other_child", "female")
    )
    for annot in no_male_from_no_child:
        assert annot.label != "child"
        assert annot.label != "male"


def test_filter_annotations_empty():
    aa_p = Path("tests/sample/test.aa")

    annots = load_annotations(aa_p)
    assert filter_annotations(annots, covered_labels=()) == []
