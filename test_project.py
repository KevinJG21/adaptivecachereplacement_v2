from cache_policies import LRUCache, LFUCache, MRUCache, ARCCache
from features import extract_features
from clustering import KMeans
from trace_generator import generate_trace
from adaptive_cache import AdaptiveCache


def test_lru():
    c = LRUCache(2)
    assert not c.access(1)
    assert not c.access(2)
    assert c.access(1)
    assert not c.access(3)
    assert not c.access(2)


def test_mru():
    c = MRUCache(2)
    c.access(1)
    c.access(2)
    assert c.access(1)
    # 1 is MRU and should be evicted after inserting 3.
    c.access(3)
    assert 1 not in c.data
    assert 3 in c.data


def test_lfu():
    c = LFUCache(2)
    c.access(1)
    c.access(2)
    c.access(1)
    c.access(3)
    assert c.access(1)
    assert not c.access(2)


def test_arc_runs():
    c = ARCCache(3)
    trace = [1, 2, 3, 1, 2, 4, 1, 2, 3, 4]
    for x in trace:
        c.access(x)


def test_features():
    f = extract_features([1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert 0 <= f["unique_ratio"] <= 1
    assert f["working_set_size"] == 3


def test_trace_lengths():
    for workload in ("sequential", "looping", "random", "mixed"):
        trace = generate_trace(workload, 137, 100, seed=7)
        assert len(trace) == 137, (workload, len(trace))
        assert all(isinstance(x, int) and x >= 0 for x in trace)


def test_switch_history():
    cache = AdaptiveCache(2, "LRU")
    cache.access(1)
    assert cache.switch_policy("LFU", position=10, reason="random")
    assert cache.switch_history[-1]["from"] == "LRU"
    assert cache.switch_history[-1]["to"] == "LFU"


def test_kmeans():
    X = [[0, 0], [0, 1], [10, 10], [10, 11]]
    km = KMeans(n_clusters=2, seed=1).fit(X)
    assert len(km.labels_) == 4
    assert len(km.cluster_centers_) == 2


if __name__ == "__main__":
    test_lru()
    test_mru()
    test_lfu()
    test_arc_runs()
    test_features()
    test_kmeans()
    test_trace_lengths()
    test_switch_history()
    print("All tests passed.")
