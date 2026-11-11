"""One-time move of the former Color Clip default onto the current one.

Every edit, roll default and sticky snapshot saved before the default changed carries the
former shipped value (1.0), and Color Clip carries onto each new frame. On a dark frame that
value takes the channel references from base noise and lamps, which tints the frame blue.
This sweeps those stores once and moves 1.0 to the current default. A value saved after
this ran is a deliberate choice and is left alone.

Best-effort and idempotent: guarded by a done flag, and it never raises into app startup.
"""

import json
import sqlite3
from contextlib import closing
from typing import Any

from negpy.features.process.models import DEFAULT_COLOR_CLIP
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_DONE_FLAG = "color_clip_default_migrated_v1"
_FIELD = "color_range_clip"
#: The value every legacy row was saved with. Fixed: the rows this sweep repairs were all written under it.
_SHIPPED_COLOR_CLIP = 1.0


def _moved(data: Any) -> bool:
    """Moves the former default in the flat dict *data*; True when it changed."""
    if not isinstance(data, dict) or _FIELD not in data:
        return False
    try:
        if float(data[_FIELD]) != _SHIPPED_COLOR_CLIP:
            return False
    except (ValueError, TypeError):
        return False
    data[_FIELD] = DEFAULT_COLOR_CLIP
    return True


def _move_rows(conn: sqlite3.Connection, table: str, key_cols: tuple[str, ...]) -> None:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
        return
    cols = ", ".join(key_cols)
    where = " AND ".join(f"{c} = ?" for c in key_cols)
    for *keys, settings_json in conn.execute(f"SELECT {cols}, settings_json FROM {table}").fetchall():
        try:
            data = json.loads(settings_json) if settings_json else None
        except (ValueError, TypeError):
            continue
        if _moved(data):
            conn.execute(f"UPDATE {table} SET settings_json = ? WHERE {where}", (json.dumps(data, default=str), *keys))


def _move_global_settings(repo) -> None:
    sticky = repo.get_global_setting("sticky_config")
    if _moved(sticky):
        repo.save_global_setting("sticky_config", sticky)
    legacy = repo.get_global_setting("last_color_range_clip")
    if legacy is not None and _moved({_FIELD: legacy}):
        repo.save_global_setting("last_color_range_clip", DEFAULT_COLOR_CLIP)
    store = repo.get_global_setting("rolls_by_id")
    if isinstance(store, dict):
        changed = [_moved(entry.get("defaults")) for entry in store.values() if isinstance(entry, dict)]
        if any(changed):
            repo.save_global_setting("rolls_by_id", store)


def migrate_color_clip_default(repo) -> None:
    """Move the former Color Clip default to the current one in saved edits, rolls and sticky settings, once."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        _move_global_settings(repo)
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            _move_rows(conn, "file_settings", ("file_hash",))
            _move_rows(conn, "edit_history", ("file_hash", "step_index"))
            _move_rows(conn, "work_prints", ("file_hash", "name"))
    except Exception:
        logger.exception("Color Clip migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)
