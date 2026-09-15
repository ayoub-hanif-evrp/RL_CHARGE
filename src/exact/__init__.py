"""Exact label-setting for frozen-route EVRPTW-GR. Timeouts are not called exact."""

from .label_setting import LabelSettingResult, solve_label_setting

__all__ = ["LabelSettingResult", "solve_label_setting"]
