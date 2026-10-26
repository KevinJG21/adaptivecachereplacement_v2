from __future__ import annotations

from cache_policies import make_policy


class AdaptiveCache:
    """
    Adaptive cache that can switch replacement policy at observation-window
    boundaries.

    Cache contents are intentionally preserved across a policy switch.
    Each policy is rebuilt from the existing cache contents using a replay
    sequence. This keeps switching semantics explicit and measurable.

    A switch only happens after a complete observation window, preventing
    per-request classifier overhead.
    """

    def __init__(self, capacity: int, initial_policy: str = "LRU"):
        self.capacity = capacity
        self.current_policy_name = initial_policy.upper()
        self.policy = make_policy(self.current_policy_name, capacity)
        self.switch_count = 0
        self.switch_history = []

    def access(self, block: int) -> bool:
        return self.policy.access(block)

    def switch_policy(self, new_policy: str, position: int, reason: str = ""):
        new_policy = new_policy.upper()
        if new_policy == self.current_policy_name:
            return False

        old_policy_name = self.current_policy_name
        old_contents = self._contents()
        self.policy = make_policy(new_policy, self.capacity)

        # Reinsert current contents. The order is from least -> most recent
        # when the old policy exposes an ordered representation; for other
        # policies, arbitrary cache order is acceptable for this simulator.
        for block in old_contents:
            self.policy.access(block)

        self.current_policy_name = new_policy
        self.switch_count += 1
        self.switch_history.append({
            "position": position,
            "from": old_policy_name,
            "to": new_policy,
            "reason": reason,
        })
        return True

    def _contents(self):
        p = self.policy
        if hasattr(p, "data"):
            if hasattr(p.data, "keys"):
                return list(p.data.keys())
            return list(p.data.keys())

        # ARC
        if hasattr(p, "t1") and hasattr(p, "t2"):
            return list(p.t1.keys()) + list(p.t2.keys())

        return []

    @property
    def name(self):
        return self.current_policy_name
