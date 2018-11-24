import pytest
from bayes_opt.bayesian_optimization import Observable


EVENTS = ["a", "b", "c"]


class SimpleObserver():
    def __init__(self):
        self.counter = 0

    def update(self, event, instance):
        self.counter += 1


def test_get_subscribers():
    observer = SimpleObserver()
    observable = Observable(events=EVENTS)

    observable.subscribe("a", observer)
    assert observer in observable.get_subscribers('a')
    assert observer not in observable.get_subscribers('b')
    assert observer not in observable.get_subscribers('c')

    assert len(observable.get_subscribers('a')) == 1)
    assert len(observable.get_subscribers('b')) == 0)
    assert len(observable.get_subscribers('c')) == 0)

# class TestObserverPattern(unittest.TestCase):
#     def setUp(self):
#         events = ['a', 'b']
#         self.observable = Observable(events)
#         self.observer = TestObserver()

#     def test_register(self):
#         self.observable.register('a', self.observer)
#         self.assertTrue(self.observer in self.observable.get_subscribers('a'))

#     def test_unregister(self):
#         self.observable.register('a', self.observer)
#         self.observable.unregister('a', self.observer)
#         self.assertTrue(self.observer not in self.observable.get_subscribers('a'))

#     def test_dispatch(self):
#         test_observer = TestObserver()
#         self.observable.register('b', test_observer)
#         self.observable.dispatch('b')
#         self.observable.dispatch('b')

#         self.assertTrue(test_observer.counter == 2)


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_observer.py
    """
    pytest.main([__file__])
