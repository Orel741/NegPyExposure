from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.tone import ToneSidebar
from negpy.features.exposure.auto_sliders import NEUTRAL
from negpy.features.process.models import ProcessMode


def test_channel_selector_retargets_and_syncs(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        exposure=replace(
            cfg.exposure,
            grade_trim_red=15.0,
            shadow_density=-0.2,
            highlight_density=0.3,
            shadow_grade=-12.0,
            highlight_grade=8.0,
            shadow_grade_trim_red=5.0,
            highlight_grade_trim_red=-3.0,
        ),
    )
    sidebar.sync_ui()

    # Global page: shared values. Darks and Lights show a density as its lightening.
    assert sidebar._curve_field("shadow_grade") == "shadow_grade"
    assert sidebar.grade_trim_slider.isHidden()
    assert abs(sidebar.shadow_density_slider.value() - 50.0) < 1e-9
    assert abs(sidebar.highlight_density_slider.value() - (-50.0)) < 1e-9
    assert abs(sidebar.shadow_grade_slider.value() - (-12.0)) < 1e-9
    assert abs(sidebar.highlight_grade_slider.value() - 8.0) < 1e-9
    assert sidebar.shadow_density_slider in sidebar._global_only
    assert sidebar.highlight_density_slider in sidebar._global_only
    # Split grade follows the channel selector (per-layer trims), not global-only.
    assert sidebar.shadow_grade_slider not in sidebar._global_only
    assert sidebar.highlight_grade_slider not in sidebar._global_only
    # Long tooltips must be rich text so Qt word-wraps them; tooltips that carry
    # their own markup (shortcut chips) must not get double-escaped. Shortcut-bearing
    # sliders get their tooltips from ControlsPanel.apply_shortcut_tooltips (single
    # source), so only locally-tooltipped widgets are asserted here.
    assert sidebar.grade_trim_slider.toolTip().startswith("<qt>")
    assert "&lt;" not in sidebar.grade_trim_slider.toolTip()

    # Red page: sliders retarget to the red trims; global-only controls grey out.
    sidebar.ch_btn.setCurrentIndex(1)

    assert sidebar._curve_field("shadow_grade") == "shadow_grade_trim_red"
    assert sidebar._curve_field("highlight_grade") == "highlight_grade_trim_red"
    assert not sidebar.grade_trim_slider.isHidden()
    assert sidebar.grade_trim_slider.value() == 15.0
    assert abs(sidebar.shadow_grade_slider.value() - 5.0) < 1e-9
    assert abs(sidebar.highlight_grade_slider.value() - (-3.0)) < 1e-9
    assert sidebar.shadow_grade_slider.label.text() == "Shadows Grade R"
    assert sidebar.ch_btn._edited[1]
    for w in sidebar._global_only:
        assert not w.isEnabled()

    # Back to Global: values and enablement restore.
    sidebar.ch_btn.setCurrentIndex(0)
    assert abs(sidebar.shadow_grade_slider.value() - (-12.0)) < 1e-9
    for w in sidebar._global_only:
        assert w.isEnabled()


def test_channel_selector_hidden_in_bw(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    sidebar.sync_ui()
    assert not sidebar.ch_btn.isHidden()
    sidebar.ch_btn.setCurrentIndex(1)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    assert sidebar.ch_btn.isHidden()
    # Forced back to the Global page.
    assert sidebar._channel_index() == 0


def test_auto_density_grade_stay_on_a_raw_slide_and_a_positive(qapp):
    """Both meter a slide on the transfer curve (transfer_auto_terms), raw or Positive."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=False))
    sidebar.sync_ui()
    assert not sidebar.auto_btn.isHidden()

    controller.state.config = replace(controller.state.config, process=replace(controller.state.config.process, positive_source=True))
    sidebar.sync_ui()
    assert not sidebar.auto_btn.isHidden()


def test_hold_black_rides_the_auto_menu_without_lighting_it(qapp):
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg, exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False, hold_black=True)
    )
    sidebar = ToneSidebar(controller)
    sidebar.sync_ui()
    assert sidebar.hold_black_action in sidebar.auto_btn.choice_menu.actions()
    assert sidebar.hold_black_action.isChecked()
    assert not sidebar.auto_btn.isChecked()

    sidebar.update_config_section = MagicMock()
    sidebar.hold_black_action.trigger()
    assert sidebar.update_config_section.call_args.kwargs["hold_black"] is False

    controller.state.config = replace(
        controller.state.config, process=replace(controller.state.config.process, process_mode=ProcessMode.E6)
    )
    sidebar.sync_ui()
    assert not sidebar.hold_black_action.isVisible()


def _autos_on(cfg):
    return replace(cfg, exposure=replace(cfg.exposure, auto_exposure=True, auto_normalize_contrast=True))


def _metered_sidebar():
    from negpy.features.exposure.normalization import LogNegativeBounds

    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = _autos_on(controller.state.config)
    controller.state.current_file_hash = "frame"
    controller.state.auto_meters["frame"] = {
        "metered_anchor": 0.55,
        "textural_range": 1.1,
        "norm_density_range": 1.4,
        "shadow_point": 0.9,
        "highlight_point": 0.03,
        "final_bounds": LogNegativeBounds((0.2, 0.5, 0.8), (1.6, 1.9, 2.1)),
    }
    return controller, ToneSidebar(controller)


def test_simple_sliders_show_the_distance_from_neutral(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, grade=102.0, whites=0.25, blacks=-0.1))
    sidebar.sync_ui()

    assert sidebar.grade_slider.value() == pytest.approx(20.0)
    assert sidebar.whites_slider.value() == pytest.approx(25.0)
    assert sidebar.blacks_slider.value() == pytest.approx(-10.0)
    # An untouched control reads as unedited, whatever Auto meters.
    for slider in (sidebar.brightness_slider, sidebar.highlight_density_slider, sidebar.shadow_density_slider):
        assert slider.value() == 0.0
        assert not slider._edited_dot.isVisibleTo(slider)


@pytest.mark.parametrize(
    "field, position, stored",
    [
        ("grade", -100.0, 180.0),
        ("highlight_density", 30.0, -0.18),
        ("whites", -40.0, -0.4),
        ("brightness", 10.0, 0.1),
    ],
)
def test_a_simple_slider_edit_stores_the_field_value(qapp, field, position, stored):
    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    sidebar._driven_sliders()[field].valueCommitted.emit(position)

    committed = controller.apply_config.call_args.args[0].exposure
    assert getattr(committed, field) == pytest.approx(stored)
    assert committed.auto_exposure


def test_reset_on_a_simple_slider_restores_the_neutral(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, grade=130.0))
    sidebar.sync_ui()
    sidebar.grade_slider.valueCommitted.emit(sidebar.grade_slider.default_value())

    assert controller.apply_config.call_args.args[0].exposure.grade == NEUTRAL["grade"]


def test_exposure_reads_and_stores_stops(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, exposure_ev=-1.5, density=1.2))
    sidebar.sync_ui()
    assert sidebar.exposure_slider.value() == pytest.approx(-1.5)
    assert sidebar.density_slider.value() == pytest.approx(1.2)

    sidebar.exposure_slider.valueCommitted.emit(0.75)
    committed = controller.apply_config.call_args.args[0].exposure
    assert committed.exposure_ev == pytest.approx(0.75)
    assert committed.density == pytest.approx(1.2)


def test_brightness_shows_only_on_a_scan_stock(qapp):
    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    assert not sidebar.brightness_slider.isHidden()

    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, paper_profile="neutral"))
    sidebar.sync_ui()
    assert sidebar.brightness_slider.isHidden()


def test_auto_toggle_flips_only_the_flag(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, density=1.2))
    sidebar.sync_ui()

    sidebar.auto_density_action.setChecked(False)
    off = controller.apply_config.call_args.args[0]
    assert not off.exposure.auto_exposure
    assert off.exposure.density == 1.2


def test_both_autos_toggle_together(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, auto_exposure=True, auto_normalize_contrast=False))
    sidebar.sync_ui()
    assert not sidebar.auto_both_action.isChecked()

    sidebar.auto_both_action.trigger()
    on = controller.apply_config.call_args.args[0].exposure
    assert on.auto_exposure and on.auto_normalize_contrast

    controller.state.config = controller.apply_config.call_args.args[0]
    sidebar.sync_ui()
    assert sidebar.auto_both_action.isChecked()
    sidebar.auto_both_action.trigger()
    off = controller.apply_config.call_args.args[0].exposure
    assert not off.auto_exposure and not off.auto_normalize_contrast


def test_a_right_click_on_auto_turns_both_autos_on_then_off(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, auto_exposure=True, auto_normalize_contrast=False))
    sidebar.sync_ui()

    QTest.mouseClick(sidebar.auto_btn, Qt.MouseButton.RightButton)
    on = controller.apply_config.call_args.args[0].exposure
    assert on.auto_exposure and on.auto_normalize_contrast

    controller.state.config = controller.apply_config.call_args.args[0]
    sidebar.sync_ui()
    QTest.mouseClick(sidebar.auto_btn, Qt.MouseButton.RightButton)
    off = controller.apply_config.call_args.args[0].exposure
    assert not off.auto_exposure and not off.auto_normalize_contrast


def test_bake_writes_shown_values_and_turns_both_autos_off(qapp):
    from negpy.desktop.auto_sliders import shown_values

    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    shown = shown_values(controller.state.config, controller.state.auto_meters["frame"])

    sidebar.auto_bake_action.trigger()

    baked = controller.apply_config.call_args.args[0].exposure
    assert not baked.auto_exposure
    assert not baked.auto_normalize_contrast
    for field in ("density", "grade", "shadow_density", "highlight_density"):
        if field in shown:
            assert getattr(baked, field) == pytest.approx(shown[field], abs=0.005)


def test_bake_item_enabled_when_an_auto_is_on(qapp):
    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    assert sidebar.auto_bake_action.isEnabled()


def test_bake_item_disabled_when_both_autos_off(qapp):
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = replace(
        controller.state.config,
        exposure=replace(controller.state.config.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )
    sidebar = ToneSidebar(controller)
    sidebar.sync_ui()
    assert not sidebar.auto_bake_action.isEnabled()


def test_confirmed_target_change_warns_but_no_change_does_not(qapp, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from negpy.desktop.view.sidebar import tone as tone_mod
    from negpy.features.exposure.models import TUNABLE_TARGETS

    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    sidebar._targets_snapshot = {k: 1.0 for k in TUNABLE_TARGETS}

    warned = []
    monkeypatch.setattr(tone_mod, "warn_targets_affect_unbaked", lambda parent, repo: warned.append(parent))
    monkeypatch.setattr(tone_mod, "apply_targets", lambda values: None)

    class _Changed:
        def values(self):
            return dict(sidebar._targets_snapshot, anchor_target_density=2.0)

    sidebar._on_targets_finished(_Changed(), int(QDialog.DialogCode.Accepted))
    assert len(warned) == 1

    warned.clear()

    class _Same:
        def values(self):
            return dict(sidebar._targets_snapshot)

    sidebar._on_targets_finished(_Same(), int(QDialog.DialogCode.Accepted))
    assert len(warned) == 0


def test_flat_negative_shows_the_softest_grade_hint(qapp):
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = _autos_on(controller.state.config)
    controller.state.current_file_hash = "flat"
    sidebar = ToneSidebar(controller)
    sidebar.sync_ui()
    assert sidebar.grade_softest_hint.isHidden()

    flat = {"metered_anchor": 0.507, "textural_range": 0.221, "norm_density_range": 0.509, "shadow_point": 1.062, "highlight_point": 0.15}
    controller.state.auto_meters["flat"] = flat
    sidebar.sync_ui()
    assert sidebar.grade_softest_hint.isHidden()

    cfg = controller.state.config
    controller.state.config = replace(cfg, exposure=replace(cfg.exposure, grade=170.0))
    sidebar.sync_ui()
    assert not sidebar.grade_softest_hint.isHidden()


def test_the_tone_profile_writes_its_key_and_waits_on_a_scan_stock(qapp):
    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    assert sidebar.tone_profile_combo.isEnabled()
    sidebar.tone_profile_combo.setCurrentIndex(sidebar.tone_profile_combo.findData("deep"))
    assert controller.apply_config.call_args.args[0].exposure.tone_profile == "deep"

    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, paper_profile="neutral", tone_profile="hard"))
    sidebar.sync_ui()
    assert not sidebar.tone_profile_combo.isEnabled()
    assert sidebar.tone_profile_combo.currentData() == "hard"
