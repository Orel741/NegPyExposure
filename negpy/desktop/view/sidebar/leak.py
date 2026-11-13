from negpy.kernel.system.i18n import tr

from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.session import ToolMode
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, clearable_tool_toggle, field_row, hint_label
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.leak.models import LEAK_SIDES


class LeakSidebar(BaseSidebar):
    """Light Leak: a frame of points on the clear film around the picture measures the leak,
    and the leak comes out of the whole frame."""

    def _init_ui(self) -> None:
        conf = self.state.config.leak
        row = QHBoxLayout()
        self.frame_btn = clearable_tool_toggle(
            "mdi6.vector-square",
            tr("Measure Leak"),
            tr("Measure Leak: drag the frame points onto the clear film around the picture, where the leak shows."),
            tr("Clear Leak: remove the measuring frame and the correction."),
        )
        row.addWidget(self.frame_btn, 1)
        self.enable_btn = self._small_toggle(
            "fa5s.lightbulb",
            "",
            conf.leak_apply,
            tr("Apply Light Leak: take the measured leak out of this frame."),
        )
        self.enable_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        row.addWidget(self.enable_btn)
        self.layout.addLayout(row)

        self.sides_btn = SegmentedChoice(
            (
                ("mdi6.vector-square", tr("All")),
                ("fa5s.arrows-alt-v", tr("Top/Bottom")),
                ("fa5s.arrows-alt-h", tr("Left/Right")),
            ),
            tr("Which sides of the measuring frame read the leak."),
            data=LEAK_SIDES,
        )
        self.sides_btn.set_choice_tooltip(0, tr("All: the four sides read the leak, for a leak that reaches the frame from several sides."))
        self.sides_btn.set_choice_tooltip(
            1, tr("Top/Bottom: only the top and bottom sides read the leak, for a leak that runs down the frame.")
        )
        self.sides_btn.set_choice_tooltip(
            2, tr("Left/Right: only the left and right sides read the leak, for a leak that runs across the frame.")
        )
        self.layout.addLayout(field_row(tr("Sides"), self.sides_btn))

        self.strength_slider = CompactSlider(tr("Strength"), 0.0, 1.5, conf.leak_strength, has_neutral=False)
        self.strength_slider.setToolTip(tr("How much of the measured leak comes out of the frame."))
        self.layout.addWidget(self.strength_slider)

        self.hint = hint_label(tr("Click Measure Leak, put the points on film with no picture, then turn on the bulb."))
        self.layout.addWidget(self.hint)

    def _connect_signals(self) -> None:
        self.frame_btn.toggled.connect(lambda on: self.controller.set_active_tool(ToolMode.LEAK_FRAME if on else ToolMode.NONE))
        self.frame_btn.clear_btn.clicked.connect(self.controller.clear_leak)
        self.enable_btn.toggled.connect(self.controller.set_leak_enabled)
        self.sides_btn.currentChanged.connect(lambda _i: self.controller.set_leak(leak_sides=self.sides_btn.currentData()))
        self.strength_slider.valueChanged.connect(
            lambda v: self.controller.set_leak(persist=False, readback_metrics=False, leak_strength=v)
        )
        self.strength_slider.valueCommitted.connect(lambda v: self.controller.set_leak(leak_strength=v))
        self.controller.tool_sync_requested.connect(self.sync_ui)
        self.sync_ui()

    def sync_ui(self) -> None:
        conf = self.state.config.leak
        has_frame = bool(conf.leak_frame)
        widgets = (self.frame_btn, self.enable_btn, self.sides_btn, self.strength_slider)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.frame_btn.setChecked(self.state.active_tool == ToolMode.LEAK_FRAME)
            self.frame_btn.clear_btn.setVisible(has_frame)
            self.enable_btn.setChecked(conf.leak_apply and has_frame)
            self.enable_btn.setEnabled(has_frame)
            self.sides_btn.setCurrentIndex(max(0, self.sides_btn.findData(conf.leak_sides)))
            self.sides_btn.setEnabled(has_frame)
            self.strength_slider.setValue(conf.leak_strength)
            self.strength_slider.setEnabled(has_frame)
            self.hint.setVisible(not (has_frame and conf.leak_apply))
        finally:
            for w in widgets:
                w.blockSignals(False)
