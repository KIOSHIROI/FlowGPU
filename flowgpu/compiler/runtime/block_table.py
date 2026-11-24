"""Request-level block table over a PagedKVCache.

The table owns request -> block mapping and the append position. It does not
schedule requests; a serving loop calls ``append`` once per decoded token.
"""
from __future__ import annotations


class BlockTable:
    """Map request ids to paged KV blocks and flat cache locations."""

    def __init__(self, cache):
        self._cache = cache
        self._requests = {}

    def __contains__(self, request_id):
        return request_id in self._requests

    def add(self, request_id) -> None:
        if request_id in self._requests:
            raise ValueError("request already exists")
        self._requests[request_id] = {"blocks": [], "length": 0}

    def append(self, request_id) -> int:
        """Append one token and return its flat cache location."""
        state = self._state(request_id)
        page = self._cache.page_size
        if state["length"] % page == 0:
            state["blocks"].append(self._cache.allocate())
        block = state["blocks"][-1]
        location = self._cache.location(block, state["length"] % page)
        state["length"] += 1
        return location

    def locations(self, request_id):
        state = self._state(request_id)
        page = self._cache.page_size
        return [self._cache.location(block, offset)
                for index, block in enumerate(state["blocks"])
                for offset in range(min(page, state["length"] - index * page))]

    def length(self, request_id) -> int:
        return self._state(request_id)["length"]

    def release(self, request_id) -> None:
        state = self._state(request_id)
        for block in state["blocks"]:
            self._cache.release(block)
        del self._requests[request_id]

    def _state(self, request_id):
        try:
            return self._requests[request_id]
        except KeyError:
            raise KeyError(f"unknown request: {request_id!r}") from None


__all__ = ["BlockTable"]
