from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_WHITE = "\033[37m"
_BRIGHT_BLUE = "\033[94m"
_BRIGHT_CYAN = "\033[96m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_MAGENTA = "\033[95m"
_BRIGHT_RED = "\033[91m"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) >= 1e-3 and abs(value) < 1e3:
            return f"{value:.4f}"
        return f"{value:.2e}"
    return str(value)


def _short_metric_name(name: str) -> str:
    replacements = {
        "vae_reconstruction": "vae",
        "pearson_mean": "pearson",
        "r2_mean": "r2",
    }
    return replacements.get(name, name)


def _metric_sort_key(name: str) -> tuple[int, str, int, str]:
    base_order = {
        "loss": 0,
        "mse": 1,
        "xstart_mse": 2,
        "effect_batch_mse": 3,
        "effect_cosine": 4,
        "vae_reconstruction": 5,
        "r2_mean": 6,
        "pearson_mean": 7,
        "mmd": 8,
        "lr": 99,
    }
    match = re.fullmatch(r"(.+)_q([0-3])", name)
    if match:
        base = match.group(1)
        quartile = int(match.group(2))
        return (base_order.get(base, 50), base, quartile + 1, name)
    return (base_order.get(name, 50), name, 0, name)


def _metric_palette(name: str) -> tuple[str, str]:
    base_name = re.sub(r"_q[0-3]$", "", name)
    palettes = {
        "loss": (_BRIGHT_RED, _BRIGHT_YELLOW),
        "mse": (_BRIGHT_MAGENTA, _BRIGHT_CYAN),
        "xstart_mse": (_MAGENTA, _BRIGHT_CYAN),
        "effect_batch_mse": (_YELLOW, _BRIGHT_BLUE),
        "effect_cosine": (_YELLOW, _BRIGHT_MAGENTA),
        "vae_reconstruction": (_GREEN, _BRIGHT_GREEN),
        "r2_mean": (_BRIGHT_GREEN, _WHITE),
        "pearson_mean": (_BRIGHT_GREEN, _WHITE),
        "mmd": (_BRIGHT_YELLOW, _WHITE),
        "lr": (_BRIGHT_BLUE, _BRIGHT_MAGENTA),
    }
    return palettes.get(base_name, (_WHITE, _BRIGHT_CYAN))


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        enabled: bool = True,
        render: str = "compact",
        progress_total: int = 0,
        progress_width: int = 24,
        run_name: str = "",
        training_stage: str = "",
        training_mode: str = "",
        world_size: int = 1,
        use_color: bool = True,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.render = render
        self.progress_total = int(progress_total)
        self.progress_width = int(progress_width)
        self.run_name = run_name
        self.training_stage = training_stage
        self.training_mode = training_mode
        self.world_size = int(world_size)
        self.use_color = bool(use_color)
        self._logger = logging.getLogger(f"perturbnova.{self.output_dir}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        self._logger.propagate = False
        if self.enabled:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
            self._logger.addHandler(handler)

    def _style(self, text: str, *styles: str) -> str:
        if not self.use_color or not styles:
            return text
        return "".join(styles) + text + _RESET

    def _progress_bar(self, step: int) -> str:
        if self.progress_total <= 0:
            return ""
        ratio = min(max(step / float(self.progress_total), 0.0), 1.0)
        filled = min(self.progress_width, max(0, int(math.floor(ratio * self.progress_width))))
        if filled <= 0:
            bar = "." * self.progress_width
        elif filled >= self.progress_width:
            bar = "#" * self.progress_width
        else:
            bar = "#" * filled + ">" + "." * (self.progress_width - filled - 1)
        return f"[{bar}]"

    def _format_rich_metrics(self, metrics: dict[str, Any]) -> str:
        pieces = []
        for key, value in metrics.items():
            label = _short_metric_name(key)
            rendered = _format_value(value)
            if label in {"loss", "mse", "vae", "r2", "pearson", "mmd"}:
                pieces.append(
                    f"{self._style(label, _BOLD, _CYAN)} {self._style(rendered, _GREEN if label != 'mmd' else _YELLOW)}"
                )
            elif label == "lr":
                pieces.append(f"{self._style('lr', _BOLD, _MAGENTA)} {self._style(rendered, _MAGENTA)}")
            else:
                pieces.append(f"{self._style(label, _BOLD)} {rendered}")
        return " | ".join(pieces)

    def _format_compact_metrics(self, metrics: dict[str, Any]) -> str:
        return ", ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items())

    def _format_mapping(self, values: dict[str, Any]) -> str:
        pieces = []
        for key, value in values.items():
            key_text = self._style(str(key), _BOLD, _BRIGHT_BLUE)
            value_text = self._style(_format_value(value), _WHITE)
            pieces.append(f"{key_text} {self._style('›', _DIM, _BRIGHT_CYAN)} {value_text}")
        return f" {self._style('•', _DIM, _BRIGHT_MAGENTA)} ".join(pieces)

    def _format_metric_table(self, step: int, split: str, metrics: dict[str, Any]) -> str:
        rows: list[tuple[str, str]] = []
        rows.append(("split", split))
        if self.progress_total > 0:
            progress_pct = min(max(step / float(self.progress_total), 0.0), 1.0) * 100.0
            rows.append(("step", f"{step}/{self.progress_total} ({progress_pct:.1f}%)"))
        else:
            rows.append(("step", str(step)))
        for key in sorted(metrics.keys(), key=_metric_sort_key):
            rows.append((key, _format_value(metrics[key])))

        plain_rows = [(str(k), str(v)) for k, v in rows]
        key_width = max(len(k) for k, _ in plain_rows)
        value_width = max(len(v) for _, v in plain_rows)

        top = self._style("╭" + "─" * (key_width + 2) + "┬" + "─" * (value_width + 2) + "╮", _BRIGHT_BLUE)
        mid = self._style("├" + "─" * (key_width + 2) + "┼" + "─" * (value_width + 2) + "┤", _DIM, _BRIGHT_BLUE)
        bot = self._style("╰" + "─" * (key_width + 2) + "┴" + "─" * (value_width + 2) + "╯", _BRIGHT_BLUE)

        lines = [top]
        for index, (key, value) in enumerate(plain_rows):
            label_color, value_color = _metric_palette(key)
            label = self._style(f"{key:<{key_width}}", _BOLD, label_color)
            rendered_value = self._style(f"{value:<{value_width}}", value_color)
            lines.append(f"{self._style('│', _BRIGHT_BLUE)} {label} {self._style('│', _BRIGHT_BLUE)} {rendered_value} {self._style('│', _BRIGHT_BLUE)}")
            if index == 1:
                lines.append(mid)
        lines.append(bot)
        return "\n".join(lines)

    def info(self, message: str, tag: str = "INFO") -> None:
        if not self.enabled:
            return
        if self.render == "compact":
            self._logger.info(message)
            return
        tag_text = self._style(f"[{tag:<5}]", _BOLD, _BLUE)
        self._logger.info("%s %s", tag_text, message)

    def log_run_header(self) -> None:
        if not self.enabled or self.render != "rich":
            return
        run_message = (
            f"{self._style(self.run_name or self.output_dir.name, _BOLD)}"
            f" | stage={self.training_stage or '-'}"
            f" | mode={self.training_mode or '-'}"
            f" | render={self.render}"
            f" | world={self.world_size}"
        )
        self.info(run_message, tag="RUN")
        self.info(f"output={self.output_dir}", tag="PATH")

    def log_mapping(self, title: str, values: dict[str, Any], tag: str = "INFO") -> None:
        if not self.enabled or not values:
            return
        payload = self._format_mapping(values)
        decorated_title = self._style(title, _BOLD, _BRIGHT_MAGENTA) if self.render == "rich" else title
        self.info(f"{decorated_title} {self._style('◆', _DIM, _BRIGHT_CYAN) if self.render == 'rich' else '|'} {payload}", tag=tag)

    def log_metrics(self, step: int, split: str, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {"step": step, "split": split, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self.render == "compact":
            formatted = self._format_compact_metrics(metrics)
            self._logger.info("[%s step=%s] %s", split, step, formatted)
            return

        tag = "TRAIN" if split == "train" else split.upper()
        table = self._format_metric_table(step=step, split=split, metrics=metrics)
        self.info(table, tag=tag)

    def log_checkpoint(self, step: int, path: str | Path, kind: str = "ckpt") -> None:
        self.info(f"step {step} | saved={path}", tag=kind.upper())

    def log_resume(self, step: int, path: str | Path, kind: str = "resume") -> None:
        self.info(f"step {step} | from={path}", tag=kind.upper())
