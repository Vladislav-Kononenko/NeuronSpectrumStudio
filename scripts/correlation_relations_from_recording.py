#!/usr/bin/env python3
"""
Один запуск для записанной EEG-сессии:
1) загружает samples/timestamps;
2) выбирает окно;
3) считает матрицы Пирсона и Спирмена;
4) считает взаимные соотношения между этими матрицами.

Это удобная обёртка над логикой check_recording.py + correlation_relations.py.

Пример:
    python scripts/correlation_relations_from_recording.py \
      --samples data/sessions/session_001_samples.npy \
      --timestamps data/sessions/session_001_timestamps.npy \
      --sfreq 500 \
      --window-seconds 2 \
      --start-seconds 0 \
      --meta-json data/sessions/session_001_meta.json \
      --out data/sessions/session_001_window_relations \
      --save-csv --print-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .check_recording import (
        corrcoef_safe,
        detect_orientation,
        load_array,
        select_window_by_samples,
        spearman_rank_rows,
        summarize_timestamps,
        to_channels_samples,
    )
    from .correlation_relations import (
        EPS_DEFAULT,
        build_relation_matrices,
        build_report,
        iter_pairs,
        print_summary,
        save_outputs,
    )
    from .matrix_tools import default_channel_names, load_channel_names, save_matrix_csv
except ImportError:
    from check_recording import (
        corrcoef_safe,
        detect_orientation,
        load_array,
        select_window_by_samples,
        spearman_rank_rows,
        summarize_timestamps,
        to_channels_samples,
    )
    from correlation_relations import (
        EPS_DEFAULT,
        build_relation_matrices,
        build_report,
        iter_pairs,
        print_summary,
        save_outputs,
    )
    from matrix_tools import default_channel_names, load_channel_names, save_matrix_csv


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Считает Pearson/Spearman и взаимные соотношения между ними прямо из файла записи."
    )
    parser.add_argument("--samples", default="data/sessions/session_001_samples.npy", help="Файл .npy со сэмплами")
    parser.add_argument("--timestamps", default="data/sessions/session_001_timestamps.npy", help="Файл .npy с временными метками")
    parser.add_argument("--sfreq", type=float, default=500.0, help="Частота дискретизации, Гц")
    parser.add_argument("--window-seconds", type=float, default=2.0, help="Длина анализируемого окна, сек")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Начало окна от старта записи, сек")
    parser.add_argument("--meta-json", default="data/sessions/session_001_meta.json", help="JSON-метаданные для имён каналов")
    parser.add_argument("--out", default="data/sessions/session_001_window_relations", help="Префикс выходных файлов")
    parser.add_argument("--eps", type=float, default=EPS_DEFAULT, help="Порог около нуля для деления и сравнения знаков")
    parser.add_argument("--top-k", type=int, default=10, help="Размер топов в отчёте")
    parser.add_argument("--include-diagonal", action="store_true", help="Учитывать диагональ в сводной статистике")
    parser.add_argument("--save-csv", action="store_true", help="Сохранить матрицы в CSV")
    parser.add_argument("--save-png", action="store_true", help="Сохранить основные матрицы как PNG")
    parser.add_argument("--print-summary", action="store_true", help="Напечатать сводку")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        samples_path = Path(args.samples)
        timestamps_path = Path(args.timestamps)
        out_prefix = Path(args.out)

        samples = load_array(samples_path, "сэмплы")
        timestamps = load_array(timestamps_path, "временные метки")

        orientation = detect_orientation(samples, timestamps)
        data_cs = to_channels_samples(samples, orientation)
        n_channels, _ = data_cs.shape

        window, ts_window, start_idx, end_idx = select_window_by_samples(
            data_cs=data_cs,
            timestamps=timestamps,
            sfreq=args.sfreq,
            window_seconds=args.window_seconds,
            start_seconds=args.start_seconds,
        )

        pearson = corrcoef_safe(window)
        spearman = corrcoef_safe(spearman_rank_rows(window))
        relations = build_relation_matrices(pearson, spearman, args.eps)

        labels = load_channel_names(args.meta_json or None, n_channels)
        if not labels or len(labels) != n_channels:
            labels = default_channel_names(n_channels)

        pairs = list(iter_pairs(pearson, spearman, relations, labels, args.eps))

        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        pearson_path = out_prefix.with_name(out_prefix.name + "_pearson.npy")
        spearman_path = out_prefix.with_name(out_prefix.name + "_spearman.npy")
        np.save(pearson_path, pearson)
        np.save(spearman_path, spearman)

        if args.save_csv:
            save_matrix_csv(out_prefix.with_name(out_prefix.name + "_pearson.csv"), pearson, labels)
            save_matrix_csv(out_prefix.with_name(out_prefix.name + "_spearman.csv"), spearman, labels)

        report = build_report(
            pearson=pearson,
            spearman=spearman,
            relations=relations,
            pairs=pairs,
            include_diagonal=args.include_diagonal,
            eps=args.eps,
            top_k=args.top_k,
            pearson_path=pearson_path,
            spearman_path=spearman_path,
        )

        report["recording"] = {
            "samples_file": str(samples_path),
            "timestamps_file": str(timestamps_path),
            "raw_samples_shape": list(samples.shape),
            "detected_orientation": orientation,
            "data_channels_samples_shape": list(data_cs.shape),
            "sampling_rate_hz": args.sfreq,
            "timestamps_summary": summarize_timestamps(timestamps),
            "window": {
                "window_seconds": args.window_seconds,
                "start_seconds": args.start_seconds,
                "start_index": start_idx,
                "end_index": end_idx,
                "window_shape": list(window.shape),
                "window_duration_from_timestamps": None
                if ts_window is None or ts_window.size < 2
                else float(ts_window[-1] - ts_window[0]),
            },
        }

        save_outputs(out_prefix, relations, report, pairs, labels, args.save_csv, args.save_png)
        print(f"Сохранена матрица Пирсона : {pearson_path}")
        print(f"Сохранена матрица Спирмена: {spearman_path}")

        if args.print_summary:
            print_summary(report, pairs, args.top_k)

        return 0
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка в correlation_relations_from_recording.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
