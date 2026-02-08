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
    """Cache key for the `(a, b, g)` optical parameter triple.

    Attributes:
        a: Single-scattering albedo.
        b: Optical thickness.
        g: Scattering anisotropy.
    """

    a: float
    b: float
    g: float

    def __hash__(self) -> int:  # Python calls this automatically
        """Return a stable hash compatible with the C cache implementation.

        Returns:
            A 64-bit FNV-1a hash for `(a, b, g)`.
        """
        return fnv1a_hash(self.a, self.b, self.g)

    def __iter__(self):
        """Iterate over key components in `(a, b, g)` order.

        Yields:
            The `a`, `b`, and `g` values.
        """
        yield from (self.a, self.b, self.g)


class Cache:
    """Dictionary-backed cache mapping `(a, b, g)` to reflectance/transmittance.

    Cached values are stored as `(ur1, ut1, uru, utu)` tuples and keyed by
    :class:`Key` so hash behavior matches the corresponding C implementation.
    """

    def __init__(self):
        """Initialize an empty cache."""
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
        """Insert or overwrite the cache entry for a parameter triple.

        Args:
            a: Single-scattering albedo.
            b: Optical thickness.
            g: Scattering anisotropy.
            ur1: Total reflection for collimated incidence.
            ut1: Total transmission for collimated incidence.
            uru: Total reflection for diffuse incidence.
            utu: Total transmission for diffuse incidence.
        """
        self._data[Key(a, b, g)] = (ur1, ut1, uru, utu)

    def get(self, a: float, b: float, g: float) -> Optional[Tuple[float, float, float, float]]:
        """Fetch a cached result for a parameter triple.

        Args:
            a: Single-scattering albedo.
            b: Optical thickness.
            g: Scattering anisotropy.

        Returns:
            The tuple `(ur1, ut1, uru, utu)` if present; otherwise `None`.
        """
        return self._data.get(Key(a, b, g))

    # helpers mirroring the C extras
    def __len__(self):
        """Return the number of cache entries.

        Returns:
            The current cache size.
        """
        return len(self._data)  # cache_size()

    def __iter__(
        self,
    ) -> Iterator[Tuple[float, float, float, float, float, float, float]]:
        """Iterate over cache records.

        Yields:
            Tuples of `(a, b, g, ur1, ut1, uru, utu)` for each entry.
        """
        for k, v in self._data.items():
            yield (*k, *v)

    # convenience for testing / demo
    def __repr__(self) -> str:
        """Return a concise debug representation of the cache.

        Returns:
            A string containing the cache size.
        """
        return f"Cache(n={len(self)})"
