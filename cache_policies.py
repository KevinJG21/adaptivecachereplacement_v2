from __future__ import annotations

from collections import OrderedDict


class CachePolicy:
    name = "BASE"

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Cache capacity must be > 0.")
        self.capacity = capacity

    def access(self, block: int) -> bool:
        raise NotImplementedError

    def reset(self):
        pass


class LRUCache(CachePolicy):
    name = "LRU"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.data = OrderedDict()

    def access(self, block: int) -> bool:
        hit = block in self.data
        if hit:
            self.data.move_to_end(block)
        else:
            self.data[block] = None
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)
        return hit

    def reset(self):
        self.data.clear()


class MRUCache(CachePolicy):
    name = "MRU"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.data = OrderedDict()

    def access(self, block: int) -> bool:
        hit = block in self.data
        if hit:
            self.data.move_to_end(block)
            return True

        # On an MRU miss, evict the current MRU *before* admitting the new block.
        if len(self.data) >= self.capacity:
            self.data.popitem(last=True)
        self.data[block] = None
        return False

    def reset(self):
        self.data.clear()


class LFUCache(CachePolicy):
    name = "LFU"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.data: dict[int, tuple[int, int]] = {}
        self.clock = 0

    def access(self, block: int) -> bool:
        self.clock += 1
        if block in self.data:
            freq, _ = self.data[block]
            self.data[block] = (freq + 1, self.clock)
            return True

        self.data[block] = (1, self.clock)
        if len(self.data) > self.capacity:
            # LFU; timestamp breaks ties by evicting the oldest.
            victim = min(
                self.data,
                key=lambda b: (self.data[b][0], self.data[b][1])
            )
            del self.data[victim]
        return False

    def reset(self):
        self.data.clear()
        self.clock = 0


class ARCCache(CachePolicy):
    """
    Compact ARC implementation.

    T1/T2 are resident lists; B1/B2 are ghost lists.
    p controls the target size of T1 and adapts based on ghost hits.
    """

    name = "ARC"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.p = 0.0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()

    def _replace(self, incoming):
        if self.t1 and (
            (incoming in self.b2 and len(self.t1) == int(self.p))
            or len(self.t1) > int(self.p)
        ):
            victim, _ = self.t1.popitem(last=False)
            self.b1[victim] = None
        else:
            if self.t2:
                victim, _ = self.t2.popitem(last=False)
                self.b2[victim] = None
            elif self.t1:
                victim, _ = self.t1.popitem(last=False)
                self.b1[victim] = None

        self._trim_ghosts()

    def _trim_ghosts(self):
        while len(self.b1) > self.capacity:
            self.b1.popitem(last=False)
        while len(self.b2) > self.capacity:
            self.b2.popitem(last=False)
        while len(self.t1) + len(self.t2) > self.capacity:
            if self.t1:
                self.t1.popitem(last=False)
            elif self.t2:
                self.t2.popitem(last=False)

    def access(self, block: int) -> bool:
        # Hit in T1/T2.
        if block in self.t1:
            self.t1.pop(block)
            self.t2[block] = None
            return True

        if block in self.t2:
            self.t2.move_to_end(block)
            return True

        # Ghost hit in B1: increase T1 target.
        if block in self.b1:
            delta = max(1.0, len(self.b2) / max(1, len(self.b1)))
            self.p = min(self.capacity, self.p + delta)
            self._replace(block)
            self.b1.pop(block, None)
            self.t2[block] = None
            self._trim_ghosts()
            return False

        # Ghost hit in B2: decrease T1 target.
        if block in self.b2:
            delta = max(1.0, len(self.b1) / max(1, len(self.b2)))
            self.p = max(0.0, self.p - delta)
            self._replace(block)
            self.b2.pop(block, None)
            self.t2[block] = None
            self._trim_ghosts()
            return False

        # Cold miss.
        if len(self.t1) + len(self.b1) >= self.capacity:
            if len(self.t1) < self.capacity:
                if self.b1:
                    self.b1.popitem(last=False)
                self._replace(block)
            else:
                if self.t1:
                    victim, _ = self.t1.popitem(last=False)
                    self.b1[victim] = None
        else:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.capacity * 2 and self.b2:
                self.b2.popitem(last=False)

        self.t1[block] = None
        self._trim_ghosts()
        return False

    def reset(self):
        self.p = 0.0
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()


POLICIES = {
    "LRU": LRUCache,
    "LFU": LFUCache,
    "MRU": MRUCache,
    "ARC": ARCCache,
}


def make_policy(name: str, capacity: int) -> CachePolicy:
    key = name.upper()
    if key not in POLICIES:
        raise ValueError(f"Unknown cache policy: {name}")
    return POLICIES[key](capacity)
