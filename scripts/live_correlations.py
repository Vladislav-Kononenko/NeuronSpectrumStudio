#!/usr/bin/env python3
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pylsl import StreamInlet, resolve_streams

try:
    from .matrix_tools import compare_matrices, format_matrix_text, load_channel_names, save_matrix_csv, save_matrix_heatmap, shift_matrix_rows_for_display
except ImportError:
    from matrix_tools import compare_matrices, format_matrix_text, load_channel_names, save_matrix_csv, save_matrix_heatmap, shift_matrix_rows_for_display

COMPARE_LABELS = {
    "shape": "форма",
    "max_abs_diff": "макс. модуль разницы",
    "mean_abs_diff": "средний модуль разницы",
    "off_diag_max_abs_diff": "макс. модуль разницы вне диагонали",
    "off_diag_mean_abs_diff": "средний модуль разницы вне диагонали",
    "off_diag_fraction_absdiff_le_0_01": "доля вне диагонали с |разницей| <= 0.01",
    "off_diag_fraction_absdiff_le_0_05": "доля вне диагонали с |разницей| <= 0.05",
}


def safe_call(fn, default=""):
    try:
        return fn()
    except Exception:
        return default


def format_rate(rate: float) -> str:
    if rate == 0:
        return "нерегулярная"
    return f"{rate:g} Hz"


def discover_streams(timeout: float) -> List:
    return resolve_streams(wait_time=timeout)


def choose_stream(
    streams,
    index: Optional[int] = None,
    name: Optional[str] = None,
    type_: Optional[str] = None,
):
    filtered = list(streams)

    if name is not None:
        filtered = [s for s in filtered if safe_call(s.name, "") == name]

    if type_ is not None:
        filtered = [s for s in filtered if safe_call(s.type, "") == type_]

    if not filtered:
        raise RuntimeError("Подходящие потоки не найдены.")

    if index is not None:
        if index < 0 or index >= len(filtered):
            raise IndexError(
                f"Индекс {index} вне диапазона для {len(filtered)} отфильтрованных потоков."
            )
        return filtered[index]

    if len(filtered) > 1:
        print("Найдено несколько подходящих потоков. Используется первый.")
        for i, s in enumerate(filtered):
            print(
                f"[{i}] имя={safe_call(s.name, '')} "
                f"тип={safe_call(s.type, '')} "
                f"каналы={safe_call(s.channel_count, '')} "
                f"частота={format_rate(safe_call(s.nominal_srate, 0))}"
            )
        print()

    return filtered[0]


def extract_channel_names(xml_text: str) -> List[str]:
    names = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return names

    for ch in root.findall(".//desc/channels/channel"):
        label = ch.findtext("label")
        if label:
            names.append(label.strip())

    return names


def stream_info_to_dict(info) -> dict:
    xml_text = info.as_xml()
    names = extract_channel_names(xml_text)

    return {
        "name": safe_call(info.name, ""),
        "type": safe_call(info.type, ""),
        "channel_count": safe_call(info.channel_count, 0),
        "nominal_srate": safe_call(info.nominal_srate, 0.0),
        "source_id": safe_call(info.source_id, ""),
        "uid": safe_call(info.uid, ""),
        "session_id": safe_call(info.session_id, ""),
        "hostname": safe_call(info.hostname, ""),
        "channel_names": names,
        "xml": xml_text,
    }


def rankdata_average_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def spearman_rank_rows(window_cs: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(window_cs, dtype=float)
    for ch in range(window_cs.shape[0]):
        ranked[ch] = rankdata_average_1d(window_cs[ch])
    return ranked


def corrcoef_safe(window_cs: np.ndarray) -> np.ndarray:
    return np.corrcoef(window_cs, rowvar=True)


def top_pairs(matrix: np.ndarray, top_k: int = 5):
    n = matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]
            if np.isfinite(value):
                pairs.append((abs(value), i, j, float(value)))
    pairs.sort(reverse=True, key=lambda t: t[0])
    return pairs[:top_k]


def matrix_stats(matrix: np.ndarray) -> dict:
    finite = np.isfinite(matrix)
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[off_diag_mask]
    off_diag_finite = off_diag[np.isfinite(off_diag)]

    return {
        "n_nan": int(np.isnan(matrix).sum()),
        "n_inf": int(np.isinf(matrix).sum()),
        "min": float(np.nanmin(matrix)) if np.any(finite) else None,
        "max": float(np.nanmax(matrix)) if np.any(finite) else None,
        "off_diag_mean": float(np.mean(off_diag_finite)) if off_diag_finite.size else None,
        "off_diag_std": float(np.std(off_diag_finite)) if off_diag_finite.size else None,
    }


def channel_label(channel_names: List[str], idx: int) -> str:
    if 0 <= idx < len(channel_names) and channel_names[idx]:
        return channel_names[idx]
    return f"канал[{idx}]"


class RingBuffer:
    def __init__(self, n_channels: int, max_samples: int):
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.data = np.zeros((n_channels, max_samples), dtype=float)
        self.timestamps = np.zeros(max_samples, dtype=float)
        self.index = 0
        self.filled = False

    def append(self, sample: np.ndarray, timestamp: float) -> None:
        self.data[:, self.index] = sample
        self.timestamps[self.index] = timestamp
        self.index = (self.index + 1) % self.max_samples
        if self.index == 0:
            self.filled = True

    def ready(self) -> bool:
        return self.filled

    def get_window(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.filled:
            return self.data[:, :self.index], self.timestamps[:self.index]

        idx = self.index
        data = np.concatenate((self.data[:, idx:], self.data[:, :idx]), axis=1)
        ts = np.concatenate((self.timestamps[idx:], self.timestamps[:idx]))
        return data, ts


def save_latest(
    out_prefix: Path,
    pearson: np.ndarray,
    spearman: np.ndarray,
    diff_abs: np.ndarray,
    report: dict,
    labels,
    save_csv: bool = False,
    save_png: bool = False,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pearson_display = shift_matrix_rows_for_display(pearson)
    spearman_display = shift_matrix_rows_for_display(spearman)
    diff_abs_display = shift_matrix_rows_for_display(diff_abs)

    pearson_path = out_prefix.with_name(out_prefix.name + "_pearson.npy")
    spearman_path = out_prefix.with_name(out_prefix.name + "_spearman.npy")
    diff_path = out_prefix.with_name(out_prefix.name + "_pearson_spearman_absdiff.npy")
    report_path = out_prefix.with_name(out_prefix.name + "_report.json")

    np.save(pearson_path, pearson_display)
    np.save(spearman_path, spearman_display)
    np.save(diff_path, diff_abs_display)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_csv:
        save_matrix_csv(out_prefix.with_name(out_prefix.name + "_pearson.csv"), pearson_display, labels)
        save_matrix_csv(out_prefix.with_name(out_prefix.name + "_spearman.csv"), spearman_display, labels)
        save_matrix_csv(out_prefix.with_name(out_prefix.name + "_pearson_spearman_absdiff.csv"), diff_abs_display, labels)

    if save_png:
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_pearson.png"), pearson_display, labels, "Матрица корреляции Пирсона")
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_spearman.png"), spearman_display, labels, "Матрица корреляции Спирмена")
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_pearson_spearman_absdiff.png"), diff_abs_display, labels, "Матрица |Пирсон - Спирмен|")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Вычисляет корреляции Пирсона и Спирмена в реальном времени из LSL EEG-потока."
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Таймаут поиска потоков в секундах.")
    parser.add_argument("--index", type=int, default=None, help="Индекс в списке отфильтрованных потоков.")
    parser.add_argument("--name", type=str, default=None, help="Точное имя потока.")
    parser.add_argument("--type", dest="type_", type=str, default="EEG", help="Точный тип потока (по умолчанию: EEG).")
    parser.add_argument("--sfreq", type=float, default=0.0, help="Переопределить частоту дискретизации. 0 = использовать nominal_srate потока.")
    parser.add_argument("--window-seconds", type=float, default=2.0, help="Длина окна в секундах.")
    parser.add_argument("--step-seconds", type=float, default=0.25, help="Интервал пересчёта в секундах.")
    parser.add_argument("--duration", type=float, default=0.0, help="Время работы в секундах. 0 = до Ctrl+C.")
    parser.add_argument("--pull-timeout", type=float, default=1.0, help="Таймаут для pull_sample().")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько самых сильных пар выводить.")
    parser.add_argument("--meta-json", type=str, default="", help="Необязательный путь к JSON-метаданным для загрузки имён каналов.")
    parser.add_argument("--print-matrices", action="store_true", help="Печатать матрицы Пирсона, Спирмена и |Пирсон-Спирмен| при каждом обновлении.")
    parser.add_argument("--matrix-decimals", type=int, default=3, help="Количество знаков после запятой для печати значений матрицы.")
    parser.add_argument("--matrix-max-rows", type=int, default=20, help="Максимальное число печатаемых строк матрицы.")
    parser.add_argument("--matrix-max-cols", type=int, default=20, help="Максимальное число печатаемых столбцов матрицы.")
    parser.add_argument("--save-csv", action="store_true", help="Если указан --out, дополнительно сохранять матрицы в CSV.")
    parser.add_argument("--save-png", action="store_true", help="Если указан --out, дополнительно сохранять PNG-теплокарты (нужен matplotlib).")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Необязательный префикс выходных файлов. Если задан, последние данные Пирсона, Спирмена и отчёт перезаписываются при каждом обновлении.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        streams = discover_streams(args.timeout)
        if not streams:
            print("LSL-потоки не найдены.", file=sys.stderr)
            return 2

        info = choose_stream(streams, index=args.index, name=args.name, type_=args.type_)
        meta = stream_info_to_dict(info)

        sfreq = args.sfreq if args.sfreq > 0 else float(meta["nominal_srate"])
        if sfreq <= 0:
            raise ValueError("Частота дискретизации неизвестна. Передайте --sfreq явно.")

        n_channels = int(meta["channel_count"])
        channel_names = meta["channel_names"] if meta["channel_names"] else []
        if args.meta_json:
            channel_names = load_channel_names(args.meta_json, n_channels)
        elif not channel_names:
            channel_names = load_channel_names(None, n_channels)

        window_size = int(round(args.window_seconds * sfreq))
        step_size = int(round(args.step_seconds * sfreq))
        if window_size < 2:
            raise ValueError("window-seconds слишком мал.")
        if step_size < 1:
            raise ValueError("step-seconds слишком мал.")

        print("=== ВЫБРАННЫЙ ПОТОК ===")
        print(f"имя                  : {meta['name']}")
        print(f"тип                  : {meta['type']}")
        print(f"каналы               : {n_channels}")
        print(f"частота потока       : {format_rate(meta['nominal_srate'])}")
        print(f"используемая частота : {sfreq:g} Hz")
        print(f"длина окна, сек      : {args.window_seconds}")
        print(f"размер окна          : {window_size} сэмплов")
        print(f"шаг, сек             : {args.step_seconds}")
        print(f"размер шага          : {step_size} сэмплов")
        if channel_names:
            print(f"имена каналов        : найдено {len(channel_names)}")
        else:
            print("имена каналов        : не найдены в метаданных")
        print()

        inlet = StreamInlet(info)
        buffer = RingBuffer(n_channels=n_channels, max_samples=window_size)

        received_samples = 0
        samples_since_update = 0
        wall_start = time.time()
        update_count = 0

        print("Запуск цикла расчёта корреляций в реальном времени. Для остановки нажмите Ctrl+C.\n")

        while True:
            if args.duration > 0 and (time.time() - wall_start) >= args.duration:
                print("Достигнута запрошенная длительность работы.")
                break

            sample, ts = inlet.pull_sample(timeout=args.pull_timeout)
            if sample is None:
                continue

            sample_arr = np.asarray(sample, dtype=float)
            if sample_arr.ndim != 1:
                print(f"Неожиданная форма сэмпла: {sample_arr.shape}", file=sys.stderr)
                continue
            if sample_arr.size != n_channels:
                print(
                    f"Несовпадение числа каналов в сэмпле: ожидалось {n_channels}, получено {sample_arr.size}",
                    file=sys.stderr,
                )
                continue

            buffer.append(sample_arr, float(ts))
            received_samples += 1
            samples_since_update += 1

            if not buffer.ready():
                continue

            if samples_since_update < step_size:
                continue

            samples_since_update = 0
            update_count += 1

            window_cs, ts_window = buffer.get_window()
            pearson = corrcoef_safe(window_cs)
            spearman = corrcoef_safe(spearman_rank_rows(window_cs))
            diff_abs = np.abs(pearson - spearman)

            pearson_stats = matrix_stats(pearson)
            spearman_stats = matrix_stats(spearman)
            compare_stats = compare_matrices(pearson, spearman)

            print(f"=== ОБНОВЛЕНИЕ {update_count} ===")
            print(f"получено сэмплов       : {received_samples}")
            print(f"форма окна             : {window_cs.shape}")
            if ts_window.size >= 2:
                print(f"длительность окна      : {float(ts_window[-1] - ts_window[0])}")
            print(f"внедиагональ Пирсона   : mean={pearson_stats['off_diag_mean']} std={pearson_stats['off_diag_std']}")
            print(f"внедиагональ Спирмена  : mean={spearman_stats['off_diag_mean']} std={spearman_stats['off_diag_std']}")
            print(f"сходство матриц        : mean_abs_diff={compare_stats['off_diag_mean_abs_diff']} max_abs_diff={compare_stats['off_diag_max_abs_diff']}")
            print()

            print(f"Топ-{args.top_k} пар Пирсона:")
            for _, i, j, value in top_pairs(pearson, top_k=args.top_k):
                li = channel_label(channel_names, i)
                lj = channel_label(channel_names, j)
                print(f"  {li} - {lj}: r={value:.6f}")
            print()

            print(f"Топ-{args.top_k} пар Спирмена:")
            for _, i, j, value in top_pairs(spearman, top_k=args.top_k):
                li = channel_label(channel_names, i)
                lj = channel_label(channel_names, j)
                print(f"  {li} - {lj}: rho={value:.6f}")
            print()

            print("Пирсон vs Спирмен:")
            for key, value in compare_stats.items():
                print(f"  {COMPARE_LABELS.get(key, key)}: {value}")
            print()

            if args.print_matrices:
                print("=== МАТРИЦА ПИРСОНА ===")
                print(format_matrix_text(shift_matrix_rows_for_display(pearson), channel_names, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
                print()
                print("=== МАТРИЦА СПИРМЕНА ===")
                print(format_matrix_text(shift_matrix_rows_for_display(spearman), channel_names, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
                print()
                print("=== МАТРИЦА |ПИРСОН - СПИРМЕН| ===")
                print(format_matrix_text(shift_matrix_rows_for_display(diff_abs), channel_names, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
                print()

            if args.out:
                report = {
                    "stream": {
                        "name": meta["name"],
                        "type": meta["type"],
                        "channel_count": n_channels,
                        "sampling_rate_hz": sfreq,
                        "channel_names": channel_names,
                    },
                    "runtime": {
                        "received_samples": received_samples,
                        "update_count": update_count,
                        "window_shape": list(window_cs.shape),
                        "window_duration_from_timestamps": (
                            None if ts_window.size < 2 else float(ts_window[-1] - ts_window[0])
                        ),
                    },
                    "pearson_stats": pearson_stats,
                    "spearman_stats": spearman_stats,
                    "pearson_vs_spearman": compare_stats,
                    "top_pearson_pairs": [
                        {
                            "i": i,
                            "j": j,
                            "label_i": channel_label(channel_names, i),
                            "label_j": channel_label(channel_names, j),
                            "value": value,
                        }
                        for _, i, j, value in top_pairs(pearson, top_k=args.top_k)
                    ],
                    "top_spearman_pairs": [
                        {
                            "i": i,
                            "j": j,
                            "label_i": channel_label(channel_names, i),
                            "label_j": channel_label(channel_names, j),
                            "value": value,
                        }
                        for _, i, j, value in top_pairs(spearman, top_k=args.top_k)
                    ],
                }
                save_latest(Path(args.out), pearson, spearman, diff_abs, report, channel_names, save_csv=args.save_csv, save_png=args.save_png)

        return 0

    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
    except Exception as exc:
        print(f"Ошибка в live_correlations.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
