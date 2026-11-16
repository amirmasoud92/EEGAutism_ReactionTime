"""Keyed child identifiers.

The analysis tables identify children by ``pid = "c" + lower(base32(HMAC-SHA256(key, "ds006780|subject|sub-XXXXX")))[:10]``
instead of the BIDS subject ID. Children are ordered by this identifier wherever an analysis sorts or resamples
children (permutations, bootstraps, surrogates), so seeded results depend on the key. The key (32 bytes) is private
and is not part of this repository.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

PID_PREFIX = "c"
PID_LENGTH = 10
KEY_BYTES = 32


def normalise_participant_id(pid: str | int) -> str:
    """``'10025'``, ``10025`` or ``'sub-10025'`` -> ``'sub-10025'``."""
    s = str(pid).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def pseudonym(key: bytes, participant_id: str | int) -> str:
    """``"c" + lower(base32(HMAC-SHA256(key, b"ds006780|subject|sub-XXXXX")))[:10]``."""
    pid = normalise_participant_id(participant_id)
    mac = hmac.new(bytes(key), f"ds006780|subject|{pid}".encode("ascii"), hashlib.sha256).digest()
    return PID_PREFIX + base64.b32encode(mac).decode("ascii").lower()[:PID_LENGTH]


class Pseudonymizer:
    """Callable ``subject -> pid`` for a given 32-byte key (raw bytes or a 64-character hex string)."""

    def __init__(self, key: bytes | str) -> None:
        if isinstance(key, str):
            key = bytes.fromhex(key.strip())
        if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
            raise ValueError(f"the key must be {KEY_BYTES} bytes")
        self._key = bytes(key)

    def __call__(self, participant_id: str | int) -> str:
        return pseudonym(self._key, participant_id)
