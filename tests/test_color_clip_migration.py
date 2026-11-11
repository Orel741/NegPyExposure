import json
import sqlite3
from contextlib import closing

import pytest

from negpy.features.process.models import DEFAULT_COLOR_CLIP
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.migrations.color_clip import migrate_color_clip_default


@pytest.fixture
def repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def _save_edit(repo, file_hash, clip):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO file_settings (file_hash, settings_json) VALUES (?, ?)",
            (file_hash, json.dumps({"color_range_clip": clip, "density": 1.0})),
        )


def _edit(repo, file_hash):
    with closing(sqlite3.connect(repo.edits_db_path)) as conn:
        row = conn.execute("SELECT settings_json FROM file_settings WHERE file_hash = ?", (file_hash,)).fetchone()
    return json.loads(row[0])


def test_the_former_default_moves_to_the_current_one(repo):
    _save_edit(repo, "old", 1.0)
    _save_edit(repo, "chosen", 5.0)
    repo.save_global_setting("sticky_config", {"color_range_clip": 1.0, "analysis_buffer": 0.1})
    repo.save_global_setting("rolls_by_id", {"r1": {"name": "Roll", "defaults": {"color_range_clip": 1.0}}})

    migrate_color_clip_default(repo)

    assert _edit(repo, "old") == {"color_range_clip": DEFAULT_COLOR_CLIP, "density": 1.0}
    assert _edit(repo, "chosen")["color_range_clip"] == 5.0
    assert repo.get_global_setting("sticky_config") == {"color_range_clip": DEFAULT_COLOR_CLIP, "analysis_buffer": 0.1}
    assert repo.get_global_setting("rolls_by_id")["r1"]["defaults"]["color_range_clip"] == DEFAULT_COLOR_CLIP


def test_a_value_saved_after_the_sweep_is_left_alone(repo):
    migrate_color_clip_default(repo)
    _save_edit(repo, "later", 1.0)

    migrate_color_clip_default(repo)

    assert _edit(repo, "later")["color_range_clip"] == 1.0
