"""
Light-weight (a, b, g) → (ur1, ut1, uru, utu) cache.
"""

from __future__ import annotations
from dataclasses import dataclass
import struct
from typing import Dict, Iterator, Tuple, Optional

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def _bits(f: float) -> int:
    """Return the IEEE-754 bit pattern of *f* as an unsigned 64-bit int."""
    return struct.unpack("<Q", struct.pack("<d", f))[0]


def fnv1a_hash(a: float, b: float, g: float) -> int:
    """64-bit FNV-1a hash of the 3 doubles (identical to the C code)."""
    h = _FNV_OFFSET
    for x in (a, b, g):
        h ^= _bits(x)
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF  # stay in 64-bit range
    return h


@dataclass(frozen=True, slots=True)
class Key:
    a: float
    b: float
    g: float

    def __hash__(self) -> int:  # Python calls this automatically
        return fnv1a_hash(self.a, self.b, self.g)

    def __iter__(self):
        yield from (self.a, self.b, self.g)


class Cache:
    """A dict-backed cache matching the C API (pythonised)."""

    def __init__(self):
        self._data: Dict[Key, Tuple[float, float, float, float]] = {}

    def put(
        self,
        a: float,
        b: float,
        g: float,
        ur1: float,
        ut1: float,
        uru: float,
        utu: float,
    ) -> None:
        """Insert or overwrite the record for (a,b,g)."""
        self._data[Key(a, b, g)] = (ur1, ut1, uru, utu)

    def get(self, a: float, b: float, g: float) -> Optional[Tuple[float, float, float, float]]:
        """Return the tuple (ur1, ut1, uru, utu) or None if absent."""
        return self._data.get(Key(a, b, g))

    # helpers mirroring the C extras
    __len__ = lambda self: len(self._data)  # cache_size()

    def __iter__(
        self,
    ) -> Iterator[Tuple[float, float, float, float, float, float, float]]:
        """Yield (a, b, g, ur1, ut1, uru, utu) for every entry."""
        for k, v in self._data.items():
            yield (*k, *v)

    # convenience for testing / demo
    def __repr__(self) -> str:
        return f"Cache(n={len(self)})"
