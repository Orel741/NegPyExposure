import pytest

from negpy.features.exposure.models import ExposureConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets.migrations.autos_off_default import migrate_autos_off_default


@pytest.fixture
def repo(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def test_a_new_frame_starts_with_both_autos_off():
    exposure = ExposureConfig()
    assert not exposure.auto_exposure and not exposure.auto_normalize_contrast


def test_the_carried_autos_turn_off_once(repo):
    repo.save_global_setting("sticky_config", {"auto_exposure": True, "auto_normalize_contrast": True, "density": 1.1})
    repo.save_global_setting("last_auto_exposure", True)

    migrate_autos_off_default(repo)

    assert repo.get_global_setting("sticky_config") == {"auto_exposure": False, "auto_normalize_contrast": False, "density": 1.1}
    assert repo.get_global_setting("last_auto_exposure") is False

    repo.save_global_setting("sticky_config", {"auto_exposure": True})
    migrate_autos_off_default(repo)
    assert repo.get_global_setting("sticky_config") == {"auto_exposure": True}
