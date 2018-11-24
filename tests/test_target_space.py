import pytest
import numpy as np
from bayes_opt.target_space import TargetSpace


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {'p1': (0, 1), 'p2': (1, 100)}


def test_keys_and_bounds_in_same_order():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    space = TargetSpace(target_func, pbounds)

    assert space.keys == ["p1", "p2",  "p3",  "p4"]
    assert all(space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 2, 3, 4]))


def test_params_to_array():
    space = TargetSpace(target_func, PBOUNDS)

    assert all(space.params_to_array({"p1": 2, "p2": 3}) == np.array([2, 3]))
    assert all(space.params_to_array({"p2": 2, "p1": 9}) == np.array([9, 2]))
    with pytest.raises(ValueError):
        space.params_to_array({"p2": 1})
    with pytest.raises(ValueError):
        space.params_to_array({"p2": 1, "p1": 7, "other": 4})
    with pytest.raises(ValueError):
        space.params_to_array({"other": 1})


def test_array_to_params():
    space = TargetSpace(target_func, PBOUNDS)

    assert space.array_to_params(np.array([2, 3])) == {"p1": 2, "p2": 3}
    with pytest.raises(ValueError):
        space.array_to_params(np.array([2]))
    with pytest.raises(ValueError):
        space.array_to_params(np.array([2, 3, 5]))


def test_register():
    space = TargetSpace(target_func, PBOUNDS)

    assert len(space) == 0
    # registering with dict
    space.register(params={"p1": 1, "p2": 2}, target=3)
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))

    # registering with array
    space.register(params={"p1": 5, "p2": 4}, target=9)
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))


def test_probe():
    space = TargetSpace(target_func, PBOUNDS)

    assert len(space) == 0
    # registering with dict
    space.probe(params={"p1": 1, "p2": 2})
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))

    # registering with array
    space.probe(params={"p1": 5, "p2": 4})
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))


def test_max():
    space = TargetSpace(target_func, PBOUNDS)

    assert space.max() == {}
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 5, "p2": 4})
    space.probe(params={"p1": 2, "p2": 3})
    space.probe(params={"p1": 1, "p2": 6})
    assert space.max() == {"params": {"p1": 5, "p2": 4}, "target": 9}


def test_res():
    space = TargetSpace(target_func, PBOUNDS)

    assert space.res() == []
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 5, "p2": 4})
    space.probe(params={"p1": 2, "p2": 3})
    space.probe(params={"p1": 1, "p2": 6})

    expected_res = [
        {"params":  {"p1": 1, "p2": 2}, "target": 3},
        {"params":  {"p1": 5, "p2": 4}, "target": 9},
        {"params":  {"p1": 2, "p2": 3}, "target": 5},
        {"params":  {"p1": 1, "p2": 6}, "target": 7},
    ]
    assert len(space.res()) == 4
    assert space.res() == expected_res


def test_set_bounds():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    space = TargetSpace(target_func, pbounds)

    # Ignore unknown keys
    space.set_bounds({"other": (7, 8)})
    assert all(space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 2, 3, 4]))

    # Update bounds accordingly
    space.set_bounds({"p2": (1, 8)})
    assert all(space.bounds[:, 0] == np.array([0, 1, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 8, 3, 4]))



# def test_nonunique_add():
#     # Adding non-unique values throws a KeyError
#     pbounds = {'p1': (0, 1), 'p2': (1, 100)}
#     space = TargetSpace(target_func, pbounds)
#     x = [0, 0]
#     y = 0
#     space.add_observation(x, y)

#     with pytest.raises(KeyError):
#         space.add_observation(x, y)

#     space._assert_internal_invariants(fast=False)


# def test_nonunique_observe():
#     # Simply re-observing a non-unique values returns the cached result
#     pbounds = {'p1': (0, 1), 'p2': (1, 100)}
#     space = TargetSpace(target_func, pbounds)
#     x = [0, 0]
#     y = 0
#     space.add_observation(x, y)

#     with pytest.raises(KeyError):
#         space.add_observation(x, y)

#     space._assert_internal_invariants(fast=False)


# def test_contains():
#     # Simply re-observing a non-unique values returns the cached result
#     pbounds = {'p1': (0, 1), 'p2': (1, 100)}
#     space = TargetSpace(target_func, pbounds)
#     # add 1000 random points
#     for x in space.random_points(1000):
#         space.observe_point(x)
#     # now all points should be unique, so test contains
#     space2 = TargetSpace(target_func, pbounds)
#     for x in space.X:
#         assert x not in space2
#         y = space2.observe_point(x)
#         assert x in space2
#         assert y == space2.observe_point(x)
#     space2._assert_internal_invariants(fast=False)


# @pytest.mark.parametrize("m", [0, 1, 2, 5, 20, 100])
# @pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
# def test_m_random_nd_points(m, n):
#     pbounds = {'p{}'.format(i): (0, i) for i in range(n)}
#     space = TargetSpace(target_func, pbounds, random_state=0)

#     X = space.random_points(m)
#     assert X.shape[0] == m
#     assert X.shape[1] == n

#     for i in range(n):
#         lower, upper = pbounds[space.keys[i]]
#         assert np.all(X.T[i] >= lower)
#         assert np.all(X.T[i] <= upper)


# @pytest.mark.parametrize("m", [0, 1, 2, 5, 20, 100])
# @pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
# def test_observe_m_nd_points(m, n):
#     pbounds = {'p{}'.format(i): (0, i) for i in range(n)}
#     space = TargetSpace(target_func, pbounds)
#     for x in space.random_points(m):
#         space.observe_point(x)
#         space._assert_internal_invariants(fast=False)
#     space._assert_internal_invariants(fast=False)


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    pytest.main([__file__])
