from computegraph.types import Function, local
from computegraph.utils import expand_nested_dict, get_nested_func


def test_expand_nested_dict():
    nd = {"a": {"aa": {"aaa": 0}, "ab": 1}, "b": {"ba": 2, "bb": 3}, "c": 4}

    no_parents = expand_nested_dict(nd, include_parents=False)

    expected_no_parents = {"a.aa.aaa": 0, "a.ab": 1, "b.ba": 2, "b.bb": 3, "c": 4}

    assert no_parents == expected_no_parents

    with_parents = expand_nested_dict(nd, include_parents=True)

    expected_with_parents = {
        "a": {"aa": {"aaa": 0}, "ab": 1},
        "a.aa": {"aaa": 0},
        "a.aa.aaa": 0,
        "a.ab": 1,
        "b": {"ba": 2, "bb": 3},
        "b.ba": 2,
        "b.bb": 3,
        "c": 4,
    }

    assert with_parents == expected_with_parents


def test_get_nested_func():
    def add_xy(x, y):
        return x + y

    f = Function(add_xy, [local("x"), local("y")])

    f_nested = get_nested_func(f, "nested")

    f_nested_expected = Function(add_xy, [local("nested.x"), local("nested.y")])

    assert f_nested == f_nested_expected
