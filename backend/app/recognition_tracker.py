"""
recognition_tracker.py
-----------------------
In-memory deduplication state for the face recognition pipeline.

WHY THIS EXISTS
---------------
The recognize endpoint is called on every webcam frame (~0.7 fps with 1400ms
interval).  Without deduplication every frame creates:
  - A recognition_log row  (before this fix: 3 000+ per session)
  - Potentially a security_events row (before this fix: limited only by a
    single global 15-second cooldown that didn't distinguish identities)

This module keeps a *per-identity track* in memory so that:
  1. The live bounding box updates on EVERY frame (no change to UX).
  2. A persistent recognition_log row is written ONLY on the first confirmed
     sighting (>= MIN_CONFIRM_FRAMES) of an identity in a new visit.
  3. A security_events row for an UNKNOWN person is written ONLY once per
     active track, and a new one is created only when the person has been
     ABSENT for at least REENTRY_SEC.

TRACK LIFECYCLE
---------------
                   first frame arrives
                         │
                   track created, confirm_count = 1
                         │
              confirm_count < MIN_CONFIRM_FRAMES
                         │
                   keep counting, no DB write
                         │
              confirm_count == MIN_CONFIRM_FRAMES
                         │
                   CONFIRMED  → write recognition_log + security_event (unknown)
                         │
              subsequent frames (same identity, gap < ACTIVE_TRACK_SEC)
                         │
                   update last_seen_at, NO new DB writes
                         │
              gap > ACTIVE_TRACK_SEC (identity disappeared)
                         │
                   track expires, last_seen_at frozen
                         │
              same identity appears again after REENTRY_SEC elapsed
                         │
                   NEW track (new visit) → new DB writes

THREADING
---------
The FastAPI app runs with uvicorn in a single-threaded asyncio event loop by
default.  All recognition requests are handled sequentially (one await at a
time in the same thread), so no mutex is required.  If you switch to
multi-worker mode (multiple processes), each process has its own in-memory
tracker state; that is acceptable because each physical camera typically
connects to one worker.  Shared state across processes would require Redis or
a DB-backed approach.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _Track:
    """State for one active detected identity."""

    # Canonical key: "STUDENT:Prapti", "GUEST:GST-…", or "UNKNOWN:<camera_id>"
    track_key: str

    # Monotonic timestamps (time.monotonic())
    first_seen_at: float
    last_seen_at: float

    # How many consecutive frames confirmed this identity
    confirm_count: int = 1

    # Whether a recognition_log row has been written for this track
    log_written: bool = False

    # Whether a security_events row has been written for this track
    event_written: bool = False

    # event_id of the security_events row (for unknown-alert updates)
    event_id: Optional[str] = None


class RecognitionTracker:
    """
    Per-camera, per-identity deduplication tracker.

    Parameters
    ----------
    active_track_sec : float
        A track is considered "active" (same visit) while the identity is
        seen within this many seconds of the last sighting.
    reentry_sec : float
        Once a track has been absent longer than this, the identity is
        treated as a new visit on next appearance.
    min_confirm_frames : int
        How many consecutive frames must match before a DB write is made.
    """

    def __init__(
        self,
        active_track_sec: float = 30.0,
        reentry_sec: float = 60.0,
        min_confirm_frames: int = 2,
    ) -> None:
        self.active_track_sec = active_track_sec
        self.reentry_sec = reentry_sec
        self.min_confirm_frames = min_confirm_frames

        # track_key → _Track
        self._tracks: Dict[str, _Track] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        identity_category: str,
        name: str,
        camera_id: str = "gate_webcam",
        guest_id: Optional[str] = None,
        is_gate_observation: bool = False,
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Record one frame observation of an identity.

        Returns
        -------
        (should_write_log, should_write_event, existing_event_id)

        should_write_log   : True → write one recognition_log row now.
        should_write_event : True → write one security_events row now.
        existing_event_id  : the event_id of the current active unknown track
                             (so callers can update last_seen_at on it instead
                             of creating a new row).
        """
        track_key = self._make_key(identity_category, name, camera_id, guest_id)
        now = time.monotonic()

        existing = self._tracks.get(track_key)

        if existing is None or self._is_expired(existing, now):
            # New visit or first sighting — start / restart track
            track = _Track(
                track_key=track_key,
                first_seen_at=now,
                last_seen_at=now,
                confirm_count=1,
                log_written=False,
                event_written=False,
                event_id=None,
            )
            self._tracks[track_key] = track
            self._evict_stale(now)
            existing = track
        else:
            # Active track — refresh last_seen
            existing.last_seen_at = now
            existing.confirm_count += 1

        if existing.confirm_count < self.min_confirm_frames:
            # Still accumulating confirmation frames
            return False, False, None

        should_write_log = not existing.log_written
        should_write_event = (
            not existing.event_written
            and (identity_category == "UNKNOWN_PERSON" or is_gate_observation)
        )

        if should_write_log:
            existing.log_written = True
        if should_write_event:
            existing.event_written = True

        return should_write_log, should_write_event, existing.event_id

    def set_event_id(
        self,
        identity_category: str,
        name: str,
        camera_id: str,
        event_id: str,
        guest_id: Optional[str] = None,
    ) -> None:
        """Attach the newly-created event_id to the active track."""
        track_key = self._make_key(identity_category, name, camera_id, guest_id)
        track = self._tracks.get(track_key)
        if track:
            track.event_id = event_id

    def is_active(
        self,
        identity_category: str,
        name: str,
        camera_id: str = "gate_webcam",
        guest_id: Optional[str] = None,
    ) -> bool:
        """Return True if this identity is currently in an active track."""
        track_key = self._make_key(identity_category, name, camera_id, guest_id)
        track = self._tracks.get(track_key)
        if track is None:
            return False
        return not self._is_expired(track, time.monotonic())

    def reset(self) -> None:
        """Clear all tracks (useful for testing)."""
        self._tracks.clear()

    def active_track_count(self) -> int:
        """Number of currently active (non-expired) tracks."""
        now = time.monotonic()
        return sum(1 for t in self._tracks.values() if not self._is_expired(t, now))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        identity_category: str,
        name: str,
        camera_id: str,
        guest_id: Optional[str],
    ) -> str:
        if identity_category == "REGISTERED_STUDENT":
            return f"STUDENT:{name}:{camera_id}"
        if identity_category == "APPROVED_GUEST":
            return f"GUEST:{guest_id or name}:{camera_id}"
        # UNKNOWN_PERSON and INCONCLUSIVE share a per-camera unknown slot.
        # We cannot reliably distinguish two different unknowns without
        # cross-frame embedding tracking; we treat all unknowns on one camera
        # as one track.  See LIMITATIONS in module docstring.
        return f"UNKNOWN:{camera_id}"

    def _is_expired(self, track: _Track, now: float) -> bool:
        """A track expires when the identity has been absent > reentry_sec."""
        return (now - track.last_seen_at) > self.reentry_sec

    def _evict_stale(self, now: float) -> None:
        """Remove tracks that have been absent far longer than reentry_sec."""
        cutoff = self.reentry_sec * 3
        stale = [k for k, t in self._tracks.items() if (now - t.last_seen_at) > cutoff]
        for k in stale:
            del self._tracks[k]


# ── Singleton ─────────────────────────────────────────────────────────────────

_tracker_instance: Optional[RecognitionTracker] = None


def get_tracker() -> RecognitionTracker:
    """
    Return the process-level singleton tracker, initialised with values from
    config.  Call this lazily (not at import time) so tests can override
    config values before the first use.
    """
    global _tracker_instance
    if _tracker_instance is None:
        from app.config import (
            RECOGNITION_ACTIVE_TRACK_SEC,
            RECOGNITION_REENTRY_SEC,
            RECOGNITION_MIN_CONFIRM_FRAMES,
        )
        _tracker_instance = RecognitionTracker(
            active_track_sec=RECOGNITION_ACTIVE_TRACK_SEC,
            reentry_sec=RECOGNITION_REENTRY_SEC,
            min_confirm_frames=RECOGNITION_MIN_CONFIRM_FRAMES,
        )
    return _tracker_instance


def reset_tracker() -> None:
    """Reset the singleton — used by tests to get a clean state."""
    global _tracker_instance
    _tracker_instance = None
