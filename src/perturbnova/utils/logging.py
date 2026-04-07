from __future__ import annotations

import json
import logging
import math
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

        progress_bar = self._progress_bar(step)
        if self.progress_total > 0:
            progress_pct = min(max(step / float(self.progress_total), 0.0), 1.0) * 100.0
            progress_text = (
                f"step {step:>6}/{self.progress_total:<6}"
                f" | {progress_pct:5.1f}%"
                f" | {self._style(progress_bar, _DIM)}"
            )
        else:
            progress_text = f"step {step}"
        metric_text = self._format_rich_metrics(metrics)
        tag = "TRAIN" if split == "train" else split.upper()
        self.info(f"{progress_text} | {metric_text}", tag=tag)

    def log_checkpoint(self, step: int, path: str | Path, kind: str = "ckpt") -> None:
        self.info(f"step {step} | saved={path}", tag=kind.upper())

    def log_resume(self, step: int, path: str | Path, kind: str = "resume") -> None:
        self.info(f"step {step} | from={path}", tag=kind.upper())
