"""One-time move of the carried Auto Density and Auto Grade onto the current default, off.

Both carry onto each new frame from the sticky snapshot, which holds the former default (on).
This sets them off there once, so a new frame opens with the fixed scan levels. Saved edits
keep their own values. A value saved after this ran is a deliberate choice and is left alone.

Best-effort and idempotent: guarded by a done flag, and it never raises into app startup.
"""

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_DONE_FLAG = "autos_off_default_migrated_v1"
_FIELDS = ("auto_exposure", "auto_normalize_contrast")


def migrate_autos_off_default(repo) -> None:
    """Set the carried Auto Density and Auto Grade off, once."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        sticky = repo.get_global_setting("sticky_config")
        if isinstance(sticky, dict) and any(sticky.get(f) for f in _FIELDS):
            sticky.update({f: False for f in _FIELDS if f in sticky})
            repo.save_global_setting("sticky_config", sticky)
        for field in _FIELDS:
            if repo.get_global_setting(f"last_{field}"):
                repo.save_global_setting(f"last_{field}", False)
    except Exception:
        logger.exception("Auto Density/Grade default migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)
