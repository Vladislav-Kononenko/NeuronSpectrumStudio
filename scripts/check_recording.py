#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    from .matrix_tools import compare_matrices, format_matrix_text, load_channel_names, save_matrix_csv, save_matrix_heatmap
except ImportError:
    from matrix_tools import compare_matrices, format_matrix_text, load_channel_names, save_matrix_csv, save_matrix_heatmap

TIMESTAMP_LABELS = {
    "n_timestamps": "число временных меток",
    "duration_seconds": "длительность, сек",
    "mean_dt": "средний dt",
    "median_dt": "медианный dt",
    "min_dt": "минимальный dt",
    "max_dt": "максимальный dt",
    "n_zero_dt": "число нулевых dt",
    "n_negative_dt": "число отрицательных dt",
}

MATRIX_STAT_LABELS = {
    "name": "имя",
    "shape": "форма",
    "n_nan": "число NaN",
    "n_inf": "число бесконечностей",
    "min": "минимум",
    "max": "максимум",
    "off_diag_mean": "среднее вне диагонали",
    "off_diag_std": "стд. откл. вне диагонали",
}

COMPARE_LABELS = {
    "shape": "форма",
    "max_abs_diff": "макс. модуль разницы",
    "mean_abs_diff": "средний модуль разницы",
    "off_diag_max_abs_diff": "макс. модуль разницы вне диагонали",
    "off_diag_mean_abs_diff": "средний модуль разницы вне диагонали",
    "off_diag_fraction_absdiff_le_0_01": "доля вне диагонали с |разницей| <= 0.01",
    "off_diag_fraction_absdiff_le_0_05": "доля вне диагонали с |разницей| <= 0.05",
}


def load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Файл «{name}» не найден: {path}")
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"«{name}» не является массивом NumPy: {path}")
    return arr


def detect_orientation(samples: np.ndarray, timestamps: Optional[np.ndarray]) -> str:
    """
    Определяет, имеют ли сэмплы форму:
      - (n_samples, n_channels)
      - (n_channels, n_samples)

    При наличии временных меток предпочитается форма,
    совпадающая с длиной массива меток времени.
    """
    if samples.ndim != 2:
        raise ValueError(f"Массив samples должен быть двумерным, получена форма {samples.shape}")

    rows, cols = samples.shape

    if timestamps is not None and timestamps.ndim == 1 and timestamps.size > 0:
        if rows == timestamps.size and cols != timestamps.size:
            return "samples_channels"
        if cols == timestamps.size and rows != timestamps.size:
            return "channels_samples"

    # Запасная эвристика:
    # В ЭЭГ обычно сэмплов намного больше, чем каналов.
    if rows > cols:
        return "samples_channels"
    return "channels_samples"


def to_channels_samples(samples: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "samples_channels":
        return samples.T
    if orientation == "channels_samples":
        return samples
    raise ValueError(f"Неизвестная ориентация массива: {orientation}")


def rankdata_average_1d(x: np.ndarray) -> np.ndarray:
    """
    Ранжирует данные с усреднением рангов при совпадениях.
    По смыслу эквивалентно scipy.stats.rankdata(method='average'),
    но реализовано без SciPy.
    Ранги начинаются с 1.
    """
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

        # Позиции i..j-1, номера рангов начинаются с 1.
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
    """
    Вычисляет матрицу корреляций по строкам (каналам).
    Значения NaN сохраняются там, где дисперсия равна нулю;
    позже они явно попадают в отчёт.
    """
    return np.corrcoef(window_cs, rowvar=True)


def select_window_by_samples(
    data_cs: np.ndarray,
    timestamps: Optional[np.ndarray],
    sfreq: float,
    window_seconds: float,
    start_seconds: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
    if sfreq <= 0:
        raise ValueError("Частота дискретизации должна быть положительной для выбора окна в секундах.")

    n_channels, n_samples_total = data_cs.shape
    window_size = int(round(window_seconds * sfreq))
    start_index = int(round(start_seconds * sfreq))
    end_index = start_index + window_size

    if window_size <= 1:
        raise ValueError("Окно слишком короткое: требуется минимум 2 сэмпла.")

    if start_index < 0:
        raise ValueError("start_seconds должен быть >= 0")

    if end_index > n_samples_total:
        raise ValueError(
            f"Запрошенное окно выходит за пределы доступных данных: "
            f"start_index={start_index}, end_index={end_index}, total_samples={n_samples_total}"
        )

    window = data_cs[:, start_index:end_index]
    ts_window = None
    if timestamps is not None and timestamps.ndim == 1 and timestamps.size >= end_index:
        ts_window = timestamps[start_index:end_index]

    return window, ts_window, start_index, end_index


def summarize_timestamps(timestamps: Optional[np.ndarray]) -> dict:
    if timestamps is None or timestamps.ndim != 1 or timestamps.size < 2:
        return {
            "n_timestamps": 0 if timestamps is None else int(timestamps.size),
            "duration_seconds": None,
            "mean_dt": None,
            "median_dt": None,
            "min_dt": None,
            "max_dt": None,
            "n_zero_dt": None,
            "n_negative_dt": None,
        }

    dts = np.diff(timestamps)
    return {
        "n_timestamps": int(timestamps.size),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "mean_dt": float(np.mean(dts)),
        "median_dt": float(np.median(dts)),
        "min_dt": float(np.min(dts)),
        "max_dt": float(np.max(dts)),
        "n_zero_dt": int(np.sum(dts == 0)),
        "n_negative_dt": int(np.sum(dts < 0)),
    }


def matrix_stats(matrix: np.ndarray, name: str) -> dict:
    finite = np.isfinite(matrix)
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[off_diag_mask]
    off_diag_finite = off_diag[np.isfinite(off_diag)]

    return {
        "name": name,
        "shape": list(matrix.shape),
        "n_nan": int(np.isnan(matrix).sum()),
        "n_inf": int(np.isinf(matrix).sum()),
        "min": float(np.nanmin(matrix)) if np.any(finite) else None,
        "max": float(np.nanmax(matrix)) if np.any(finite) else None,
        "off_diag_mean": float(np.mean(off_diag_finite)) if off_diag_finite.size else None,
        "off_diag_std": float(np.std(off_diag_finite)) if off_diag_finite.size else None,
    }


def top_pairs(matrix: np.ndarray, top_k: int = 10, absolute: bool = True):
    n = matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]
            if np.isfinite(value):
                score = abs(value) if absolute else value
                pairs.append((score, i, j, float(value)))

    pairs.sort(reverse=True, key=lambda t: t[0])
    return pairs[:top_k]


def save_outputs(
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

    pearson_path = out_prefix.with_name(out_prefix.name + "_pearson.npy")
    spearman_path = out_prefix.with_name(out_prefix.name + "_spearman.npy")
    diff_path = out_prefix.with_name(out_prefix.name + "_pearson_spearman_absdiff.npy")
    report_path = out_prefix.with_name(out_prefix.name + "_report.json")

    np.save(pearson_path, pearson)
    np.save(spearman_path, spearman)
    np.save(diff_path, diff_abs)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Сохранена матрица Пирсона      : {pearson_path}")
    print(f"Сохранена матрица Спирмена     : {spearman_path}")
    print(f"Сохранена матрица модулей разн.: {diff_path}")
    print(f"Сохранён отчёт                 : {report_path}")

    if save_csv:
        pearson_csv = out_prefix.with_name(out_prefix.name + "_pearson.csv")
        spearman_csv = out_prefix.with_name(out_prefix.name + "_spearman.csv")
        diff_csv = out_prefix.with_name(out_prefix.name + "_pearson_spearman_absdiff.csv")
        save_matrix_csv(pearson_csv, pearson, labels)
        save_matrix_csv(spearman_csv, spearman, labels)
        save_matrix_csv(diff_csv, diff_abs, labels)
        print(f"Сохранён CSV Пирсона          : {pearson_csv}")
        print(f"Сохранён CSV Спирмена         : {spearman_csv}")
        print(f"Сохранён CSV с разницей       : {diff_csv}")

    if save_png:
        pearson_png = out_prefix.with_name(out_prefix.name + "_pearson.png")
        spearman_png = out_prefix.with_name(out_prefix.name + "_spearman.png")
        diff_png = out_prefix.with_name(out_prefix.name + "_pearson_spearman_absdiff.png")
        save_matrix_heatmap(pearson_png, pearson, labels, "Матрица корреляции Пирсона")
        save_matrix_heatmap(spearman_png, spearman, labels, "Матрица корреляции Спирмена")
        save_matrix_heatmap(diff_png, diff_abs, labels, "Матрица |Пирсон - Спирмен|")
        print(f"Сохранён PNG Пирсона          : {pearson_png}")
        print(f"Сохранён PNG Спирмена         : {spearman_png}")
        print(f"Сохранён PNG с разницей       : {diff_png}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Загружает записанную LSL EEG-сессию и вычисляет корреляции Пирсона и Спирмена на выбранном окне."
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/sessions/session_001_samples.npy",
        help="Путь к сохранённому файлу .npy со сэмплами.",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default="data/sessions/session_001_timestamps.npy",
        help="Путь к сохранённому файлу .npy с временными метками.",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=500.0,
        help="Частота дискретизации в Гц. По умолчанию: 500.0",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="Длина окна в секундах. По умолчанию: 2.0",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Начальная позиция окна в секундах. По умолчанию: 0.0",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/sessions/session_001_check",
        help="Префикс выходных файлов без расширения.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Сколько самых сильных пар каналов вывести. По умолчанию: 10",
    )
    parser.add_argument(
        "--meta-json",
        type=str,
        default="data/sessions/session_001_meta.json",
        help="Необязательный путь к JSON-метаданным для загрузки имён каналов.",
    )
    parser.add_argument(
        "--print-matrices",
        action="store_true",
        help="Печатать в консоль матрицы Пирсона, Спирмена и |Пирсон-Спирмен|.",
    )
    parser.add_argument(
        "--matrix-decimals",
        type=int,
        default=3,
        help="Количество знаков после запятой для печати значений матрицы.",
    )
    parser.add_argument(
        "--matrix-max-rows",
        type=int,
        default=20,
        help="Максимальное число печатаемых строк матрицы.",
    )
    parser.add_argument(
        "--matrix-max-cols",
        type=int,
        default=20,
        help="Максимальное число печатаемых столбцов матрицы.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Сохранять матрицы Пирсона, Спирмена и разницы в CSV.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Сохранять матрицы Пирсона, Спирмена и разницы как PNG-теплокарты (нужен matplotlib).",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        samples_path = Path(args.samples)
        timestamps_path = Path(args.timestamps)
        out_prefix = Path(args.out)

        samples = load_array(samples_path, "сэмплы")
        timestamps = load_array(timestamps_path, "временные метки")

        orientation = detect_orientation(samples, timestamps)
        data_cs = to_channels_samples(samples, orientation)

        n_channels, n_samples_total = data_cs.shape

        print("=== ИНФОРМАЦИЯ О ЗАПИСИ ===")
        print(f"файл сэмплов              : {samples_path}")
        print(f"файл временных меток      : {timestamps_path}")
        print(f"форма исходных сэмплов    : {samples.shape}")
        print(f"определённая ориентация   : {orientation}")
        print(f"данные (каналы, сэмплы)   : {data_cs.shape}")
        print(f"частота дискретизации     : {args.sfreq} Hz")
        print()

        ts_summary = summarize_timestamps(timestamps)
        print("=== ВРЕМЕННЫЕ МЕТКИ ===")
        for key, value in ts_summary.items():
            print(f"{TIMESTAMP_LABELS.get(key, key):24}: {value}")
        print()

        window, ts_window, start_idx, end_idx = select_window_by_samples(
            data_cs=data_cs,
            timestamps=timestamps,
            sfreq=args.sfreq,
            window_seconds=args.window_seconds,
            start_seconds=args.start_seconds,
        )

        print("=== ВЫБРАННОЕ ОКНО ===")
        print(f"длина окна в секундах     : {args.window_seconds}")
        print(f"начало в секундах         : {args.start_seconds}")
        print(f"начальный индекс          : {start_idx}")
        print(f"конечный индекс           : {end_idx}")
        print(f"форма окна                : {window.shape}")
        if ts_window is not None and ts_window.size >= 2:
            print(f"длительность окна (ts)    : {float(ts_window[-1] - ts_window[0])}")
        print()

        pearson = corrcoef_safe(window)
        ranked = spearman_rank_rows(window)
        spearman = corrcoef_safe(ranked)
        diff_abs = np.abs(pearson - spearman)

        labels = load_channel_names(args.meta_json, n_channels)

        pearson_stats = matrix_stats(pearson, "Пирсон")
        spearman_stats = matrix_stats(spearman, "Спирмен")
        compare_stats = compare_matrices(pearson, spearman)

        print("=== СТАТИСТИКА МАТРИЦ ===")
        print("Пирсон:")
        for key, value in pearson_stats.items():
            print(f"  {MATRIX_STAT_LABELS.get(key, key):28}: {value}")
        print("Спирмен:")
        for key, value in spearman_stats.items():
            print(f"  {MATRIX_STAT_LABELS.get(key, key):28}: {value}")
        print()

        print(f"=== ТОП-{args.top_k} ПАР ПИРСОНА (по |r|) ===")
        for score, i, j, value in top_pairs(pearson, top_k=args.top_k, absolute=True):
            print(f"{labels[i]} - {labels[j]} : r={value:.6f}")
        print()

        print(f"=== ТОП-{args.top_k} ПАР СПИРМЕНА (по |rho|) ===")
        for score, i, j, value in top_pairs(spearman, top_k=args.top_k, absolute=True):
            print(f"{labels[i]} - {labels[j]} : rho={value:.6f}")
        print()

        print("=== ПИРСОН vs СПИРМЕН ===")
        for key, value in compare_stats.items():
            print(f"{COMPARE_LABELS.get(key, key):38}: {value}")
        print()

        if args.print_matrices:
            print("=== МАТРИЦА ПИРСОНА ===")
            print(format_matrix_text(pearson, labels, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
            print()
            print("=== МАТРИЦА СПИРМЕНА ===")
            print(format_matrix_text(spearman, labels, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
            print()
            print("=== МАТРИЦА |ПИРСОН - СПИРМЕН| ===")
            print(format_matrix_text(diff_abs, labels, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
            print()

        report = {
            "recording": {
                "samples_file": str(samples_path),
                "timestamps_file": str(timestamps_path),
                "raw_samples_shape": list(samples.shape),
                "detected_orientation": orientation,
                "data_channels_samples_shape": list(data_cs.shape),
                "sampling_rate_hz": args.sfreq,
            },
            "timestamps": ts_summary,
            "window": {
                "window_seconds": args.window_seconds,
                "start_seconds": args.start_seconds,
                "start_index": start_idx,
                "end_index": end_idx,
                "window_shape": list(window.shape),
                "window_duration_from_timestamps": (
                    None if ts_window is None or ts_window.size < 2 else float(ts_window[-1] - ts_window[0])
                ),
            },
            "pearson": pearson_stats,
            "spearman": spearman_stats,
            "pearson_vs_spearman": compare_stats,
            "top_pearson_pairs": [
                {"i": i, "j": j, "label_i": labels[i], "label_j": labels[j], "value": value} for _, i, j, value in top_pairs(pearson, top_k=args.top_k)
            ],
            "top_spearman_pairs": [
                {"i": i, "j": j, "label_i": labels[i], "label_j": labels[j], "value": value} for _, i, j, value in top_pairs(spearman, top_k=args.top_k)
            ],
        }

        save_outputs(out_prefix, pearson, spearman, diff_abs, report, labels, save_csv=args.save_csv, save_png=args.save_png)
        return 0

    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка в check_recording.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
