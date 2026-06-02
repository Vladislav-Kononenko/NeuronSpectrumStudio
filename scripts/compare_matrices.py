#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .matrix_tools import compare_matrices, format_matrix_text, load_channel_names, save_matrix_csv, save_matrix_heatmap, shift_matrix_rows_for_display
except ImportError:
    from matrix_tools import compare_matrices, format_matrix_text, load_channel_names, save_matrix_csv, save_matrix_heatmap, shift_matrix_rows_for_display


def load_matrix(prefix: Path, suffix: str) -> np.ndarray:
    path = prefix.with_name(prefix.name + suffix)
    if not path.exists():
        raise FileNotFoundError(f"Файл матрицы не найден: {path}")
    return np.load(path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Сравнивает два сохранённых набора матриц, например офлайн- и онлайн-результаты."
    )
    parser.add_argument("--reference-prefix", required=True, help="Префикс эталонных файлов без суффикса.")
    parser.add_argument("--candidate-prefix", required=True, help="Префикс сравниваемых файлов без суффикса.")
    parser.add_argument("--meta-json", default="", help="Необязательный путь к JSON-метаданным для загрузки имён каналов.")
    parser.add_argument("--out", default="data/sessions/matrix_compare", help="Префикс выходных файлов без расширения.")
    parser.add_argument("--print-matrices", action="store_true", help="Печатать матрицы модулей разности.")
    parser.add_argument("--matrix-decimals", type=int, default=3)
    parser.add_argument("--matrix-max-rows", type=int, default=20)
    parser.add_argument("--matrix-max-cols", type=int, default=20)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        ref_prefix = Path(args.reference_prefix)
        cand_prefix = Path(args.candidate_prefix)
        out_prefix = Path(args.out)

        pearson_ref = load_matrix(ref_prefix, "_pearson.npy")
        pearson_cand = load_matrix(cand_prefix, "_pearson.npy")
        spearman_ref = load_matrix(ref_prefix, "_spearman.npy")
        spearman_cand = load_matrix(cand_prefix, "_spearman.npy")

        if pearson_ref.shape != pearson_cand.shape or spearman_ref.shape != spearman_cand.shape:
            raise ValueError("Формы эталонных и сравниваемых матриц должны совпадать.")

        labels = load_channel_names(args.meta_json or None, pearson_ref.shape[0])

        pearson_diff_abs = np.abs(pearson_ref - pearson_cand)
        spearman_diff_abs = np.abs(spearman_ref - spearman_cand)
        pearson_diff_abs_display = shift_matrix_rows_for_display(pearson_diff_abs)
        spearman_diff_abs_display = shift_matrix_rows_for_display(spearman_diff_abs)

        pearson_stats = compare_matrices(pearson_ref, pearson_cand)
        spearman_stats = compare_matrices(spearman_ref, spearman_cand)

        print("=== ПИРСОН: эталон vs сравниваемый ===")
        for key, value in pearson_stats.items():
            print(f"{key:30}: {value}")
        print()

        print("=== СПИРМЕН: эталон vs сравниваемый ===")
        for key, value in spearman_stats.items():
            print(f"{key:30}: {value}")
        print()

        if args.print_matrices:
            print("=== |ПИРСОН_ЭТАЛОН - ПИРСОН_СРАВН| ===")
            print(format_matrix_text(pearson_diff_abs_display, labels, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
            print()
            print("=== |СПИРМЕН_ЭТАЛОН - СПИРМЕН_СРАВН| ===")
            print(format_matrix_text(spearman_diff_abs_display, labels, args.matrix_decimals, args.matrix_max_rows, args.matrix_max_cols))
            print()

        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_prefix.with_name(out_prefix.name + "_pearson_absdiff.npy"), pearson_diff_abs_display)
        np.save(out_prefix.with_name(out_prefix.name + "_spearman_absdiff.npy"), spearman_diff_abs_display)
        report = {
            "reference_prefix": str(ref_prefix),
            "candidate_prefix": str(cand_prefix),
            "pearson": pearson_stats,
            "spearman": spearman_stats,
        }
        out_prefix.with_name(out_prefix.name + "_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.save_csv:
            save_matrix_csv(out_prefix.with_name(out_prefix.name + "_pearson_absdiff.csv"), pearson_diff_abs_display, labels)
            save_matrix_csv(out_prefix.with_name(out_prefix.name + "_spearman_absdiff.csv"), spearman_diff_abs_display, labels)

        if args.save_png:
            save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_pearson_absdiff.png"), pearson_diff_abs_display, labels, "|Пирсон: эталон - сравниваемый|")
            save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_spearman_absdiff.png"), spearman_diff_abs_display, labels, "|Спирмен: эталон - сравниваемый|")

        return 0
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка в compare_matrices.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
