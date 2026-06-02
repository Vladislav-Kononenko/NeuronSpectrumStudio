from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from pylsl import StreamInlet

from scripts import live_correlations, lsl_record, matrix_tools

from . import services


class RecordingWorker(QThread):
    log = Signal(str)
    stats = Signal(dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        descriptor: services.StreamDescriptor,
        out_prefix: str,
        discover_timeout: float = 2.0,
        pull_timeout: float = 0.5,
        max_duration: float = 0.0,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.out_prefix = out_prefix
        self.discover_timeout = discover_timeout
        self.pull_timeout = pull_timeout
        self.max_duration = max_duration
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            info = services.resolve_stream(self.descriptor, self.discover_timeout)
            meta = lsl_record.stream_info_to_dict(info)
            inlet = StreamInlet(info)

            samples: list[list[float]] = []
            timestamps: list[float] = []
            wall_start = time.time()
            last_emit = wall_start

            self.log.emit(f"Запись начата: {meta['name']} ({meta['type']})")
            while not self._stop_requested:
                if self.max_duration > 0 and (time.time() - wall_start) >= self.max_duration:
                    self.log.emit("Достигнута заданная длительность записи.")
                    break

                sample, ts = inlet.pull_sample(timeout=self.pull_timeout)
                if sample is None:
                    continue

                samples.append(sample)
                timestamps.append(float(ts))

                now = time.time()
                if now - last_emit >= 0.5:
                    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
                    self.stats.emit(
                        {
                            "n_samples": len(samples),
                            "n_channels": len(sample),
                            "duration_seconds": float(duration),
                            "nominal_srate": float(meta["nominal_srate"]),
                        }
                    )
                    last_emit = now

            if not samples:
                raise RuntimeError("Не получено ни одного сэмпла. Сохранение не выполнено.")

            samples_array = np.asarray(samples, dtype=float)
            timestamps_array = np.asarray(timestamps, dtype=float)
            summary = lsl_record.compute_summary(samples_array, timestamps_array, float(meta["nominal_srate"]))
            lsl_record.save_outputs(samples_array, timestamps_array, meta, summary, Path(self.out_prefix))

            result = {
                "out_prefix": self.out_prefix,
                "samples_path": str(Path(self.out_prefix).with_name(Path(self.out_prefix).name + "_samples.npy")),
                "timestamps_path": str(Path(self.out_prefix).with_name(Path(self.out_prefix).name + "_timestamps.npy")),
                "meta_path": str(Path(self.out_prefix).with_name(Path(self.out_prefix).name + "_meta.json")),
                "summary": summary,
            }
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class OfflineAnalysisWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.log.emit("Запущен offline-анализ записи.")
            result = services.analyze_recording(**self.kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class CompareWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.log.emit("Запущено сравнение наборов матриц.")
            result = services.compare_result_sets(**self.kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class LiveCorrelationWorker(QThread):
    log = Signal(str)
    update_ready = Signal(dict)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(
        self,
        descriptor: services.StreamDescriptor,
        discover_timeout: float,
        sfreq_override: float,
        window_seconds: float,
        step_seconds: float,
        pull_timeout: float,
        meta_json: str | None = None,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.discover_timeout = discover_timeout
        self.sfreq_override = sfreq_override
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.pull_timeout = pull_timeout
        self.meta_json = meta_json
        self.top_k = top_k
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            info = services.resolve_stream(self.descriptor, self.discover_timeout)
            meta = live_correlations.stream_info_to_dict(info)

            sfreq = self.sfreq_override if self.sfreq_override > 0 else float(meta["nominal_srate"])
            if sfreq <= 0:
                raise RuntimeError("Частота потока не определена. Укажите её вручную.")

            n_channels = int(meta["channel_count"])
            labels = meta["channel_names"] if meta["channel_names"] else []
            if self.meta_json:
                labels = matrix_tools.load_channel_names(self.meta_json, n_channels)
            labels = services.safe_channel_labels(labels, n_channels)

            window_size = int(round(self.window_seconds * sfreq))
            step_size = int(round(self.step_seconds * sfreq))
            if window_size < 2:
                raise RuntimeError("Окно слишком маленькое.")
            if step_size < 1:
                raise RuntimeError("Шаг обновления слишком маленький.")

            inlet = StreamInlet(info)
            buffer = live_correlations.RingBuffer(n_channels=n_channels, max_samples=window_size)
            received_samples = 0
            samples_since_update = 0
            update_count = 0

            self.log.emit(f"Live-анализ запущен: {meta['name']} ({sfreq:g} Гц)")
            while not self._stop_requested:
                sample, ts = inlet.pull_sample(timeout=self.pull_timeout)
                if sample is None:
                    continue

                sample_arr = np.asarray(sample, dtype=float)
                if sample_arr.size != n_channels:
                    self.log.emit(f"Пропущен сэмпл с неожиданным числом каналов: {sample_arr.size}")
                    continue

                buffer.append(sample_arr, float(ts))
                received_samples += 1
                samples_since_update += 1

                if not buffer.ready() or samples_since_update < step_size:
                    continue

                samples_since_update = 0
                update_count += 1

                window_cs, ts_window = buffer.get_window()
                pearson = live_correlations.corrcoef_safe(window_cs)
                spearman = live_correlations.corrcoef_safe(live_correlations.spearman_rank_rows(window_cs))
                diff_abs = np.abs(pearson - spearman)

                pearson_stats = live_correlations.matrix_stats(pearson)
                spearman_stats = live_correlations.matrix_stats(spearman)
                compare_stats = matrix_tools.compare_matrices(pearson, spearman)

                report = {
                    "stream": {
                        "name": meta["name"],
                        "type": meta["type"],
                        "channel_count": n_channels,
                        "sampling_rate_hz": sfreq,
                        "channel_names": labels,
                    },
                    "runtime": {
                        "received_samples": received_samples,
                        "update_count": update_count,
                        "window_shape": list(window_cs.shape),
                        "window_duration_from_timestamps": None if ts_window.size < 2 else float(ts_window[-1] - ts_window[0]),
                    },
                    "pearson_stats": pearson_stats,
                    "spearman_stats": spearman_stats,
                    "pearson_vs_spearman": compare_stats,
                    "top_pearson_pairs": [
                        {
                            "i": i,
                            "j": j,
                            "label_i": labels[i],
                            "label_j": labels[j],
                            "value": value,
                        }
                        for _, i, j, value in live_correlations.top_pairs(pearson, top_k=self.top_k)
                    ],
                    "top_spearman_pairs": [
                        {
                            "i": i,
                            "j": j,
                            "label_i": labels[i],
                            "label_j": labels[j],
                            "value": value,
                        }
                        for _, i, j, value in live_correlations.top_pairs(spearman, top_k=self.top_k)
                    ],
                }

                self.update_ready.emit(
                    {
                        "labels": labels,
                        "pearson": pearson,
                        "spearman": spearman,
                        "diff_abs": diff_abs,
                        "report": report,
                        "stats": {
                            "received_samples": received_samples,
                            "update_count": update_count,
                            "window_shape": list(window_cs.shape),
                            "window_duration_seconds": None if ts_window.size < 2 else float(ts_window[-1] - ts_window[0]),
                            "pearson_off_diag_mean": pearson_stats["off_diag_mean"],
                            "spearman_off_diag_mean": spearman_stats["off_diag_mean"],
                            "matrix_diff_mean": compare_stats["off_diag_mean_abs_diff"],
                        },
                    }
                )

            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
