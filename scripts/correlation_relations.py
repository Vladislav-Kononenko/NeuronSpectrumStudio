#!/usr/bin/env python3
"""
Расчёт взаимных соотношений между матрицами корреляций Пирсона и Спирмена.

Скрипт работает с уже сохранёнными файлами:
    <prefix>_pearson.npy
    <prefix>_spearman.npy

На выходе сохраняет матрицы:
    <out>_signed_diff.npy        = Pearson - Spearman
    <out>_absdiff.npy            = |Pearson - Spearman|
    <out>_signed_ratio.npy       = Pearson / Spearman
    <out>_abs_ratio.npy          = |Pearson| / |Spearman|
    <out>_mean_abs_strength.npy  = (|Pearson| + |Spearman|) / 2
    <out>_sign_agreement.npy     = 1, если знаки совпали; 0, если различаются; NaN около нуля
    <out>_pairs.csv              = таблица всех пар каналов i < j
    <out>_report.json            = сводный отчёт

Пример:
    python scripts/correlation_relations.py \
      --prefix data/sessions/session_001_check \
      --meta-json data/sessions/session_001_meta.json \
      --out data/sessions/session_001_relations \
      --save-csv --save-png --print-summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from .matrix_tools import (
        default_channel_names,
        format_matrix_text,
        load_channel_names,
        save_matrix_csv,
        save_matrix_heatmap,
    )
except ImportError:
    from matrix_tools import (
        default_channel_names,
        format_matrix_text,
        load_channel_names,
        save_matrix_csv,
        save_matrix_heatmap,
    )


EPS_DEFAULT = 1e-12


def load_matrix(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Файл матрицы {name} не найден: {path}")
    matrix = np.load(path)
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} не является NumPy-массивом: {path}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} должна быть квадратной 2D-матрицей, получена форма {matrix.shape}")
    return matrix.astype(float, copy=False)


def load_pair_from_prefix(prefix: Path) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
    pearson_path = prefix.with_name(prefix.name + "_pearson.npy")
    spearman_path = prefix.with_name(prefix.name + "_spearman.npy")
    pearson = load_matrix(pearson_path, "Пирсон")
    spearman = load_matrix(spearman_path, "Спирмен")
    return pearson, spearman, pearson_path, spearman_path


def validate_same_shape(pearson: np.ndarray, spearman: np.ndarray) -> None:
    if pearson.shape != spearman.shape:
        raise ValueError(f"Формы матриц различаются: Pearson {pearson.shape}, Spearman {spearman.shape}")


def offdiag_mask(n: int, include_diagonal: bool = False) -> np.ndarray:
    if include_diagonal:
        return np.ones((n, n), dtype=bool)
    return ~np.eye(n, dtype=bool)


def upper_triangle_values(matrix: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    k = 0 if include_diagonal else 1
    idx = np.triu_indices_from(matrix, k=k)
    values = matrix[idx]
    return values[np.isfinite(values)]


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=float)
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > eps)
    result[mask] = numerator[mask] / denominator[mask]
    return result


def sign_agreement_matrix(pearson: np.ndarray, spearman: np.ndarray, eps: float) -> np.ndarray:
    result = np.full_like(pearson, np.nan, dtype=float)
    p_nonzero = np.abs(pearson) > eps
    s_nonzero = np.abs(spearman) > eps
    mask = np.isfinite(pearson) & np.isfinite(spearman) & p_nonzero & s_nonzero
    result[mask] = (np.sign(pearson[mask]) == np.sign(spearman[mask])).astype(float)
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    av = a[mask]
    bv = b[mask]
    if av.size == 0:
        return None
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return None
    return float(np.dot(av, bv) / denom)


def vector_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    av = a[mask]
    bv = b[mask]
    if av.size < 2:
        return None
    if np.std(av) == 0.0 or np.std(bv) == 0.0:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def fraction(condition: np.ndarray) -> Optional[float]:
    finite = np.isfinite(condition)
    if not np.any(finite):
        return None
    return float(np.mean(condition[finite]))


def matrix_basic_stats(matrix: np.ndarray, mask: np.ndarray) -> dict:
    values = matrix[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def build_relation_matrices(pearson: np.ndarray, spearman: np.ndarray, eps: float) -> dict:
    signed_diff = pearson - spearman
    absdiff = np.abs(signed_diff)
    signed_ratio = safe_divide(pearson, spearman, eps)
    abs_ratio = safe_divide(np.abs(pearson), np.abs(spearman), eps)
    mean_abs_strength = (np.abs(pearson) + np.abs(spearman)) / 2.0
    sign_agreement = sign_agreement_matrix(pearson, spearman, eps)
    abs_dominance = np.abs(pearson) - np.abs(spearman)

    return {
        "signed_diff": signed_diff,
        "absdiff": absdiff,
        "signed_ratio": signed_ratio,
        "abs_ratio": abs_ratio,
        "mean_abs_strength": mean_abs_strength,
        "sign_agreement": sign_agreement,
        "abs_dominance": abs_dominance,
    }


def iter_pairs(
    pearson: np.ndarray,
    spearman: np.ndarray,
    relations: dict,
    labels: List[str],
    eps: float,
) -> Iterable[dict]:
    n = pearson.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            p = pearson[i, j]
            s = spearman[i, j]
            if abs(p) <= eps and abs(s) <= eps:
                sign_relation = "оба около нуля"
            elif abs(p) <= eps or abs(s) <= eps:
                sign_relation = "один около нуля"
            elif np.sign(p) == np.sign(s):
                sign_relation = "знак совпадает"
            else:
                sign_relation = "знак различается"

            yield {
                "i": i,
                "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "pearson": float(p) if np.isfinite(p) else None,
                "spearman": float(s) if np.isfinite(s) else None,
                "signed_diff": float(relations["signed_diff"][i, j]) if np.isfinite(relations["signed_diff"][i, j]) else None,
                "absdiff": float(relations["absdiff"][i, j]) if np.isfinite(relations["absdiff"][i, j]) else None,
                "signed_ratio": float(relations["signed_ratio"][i, j]) if np.isfinite(relations["signed_ratio"][i, j]) else None,
                "abs_ratio": float(relations["abs_ratio"][i, j]) if np.isfinite(relations["abs_ratio"][i, j]) else None,
                "mean_abs_strength": float(relations["mean_abs_strength"][i, j]) if np.isfinite(relations["mean_abs_strength"][i, j]) else None,
                "abs_dominance": float(relations["abs_dominance"][i, j]) if np.isfinite(relations["abs_dominance"][i, j]) else None,
                "sign_agreement": float(relations["sign_agreement"][i, j]) if np.isfinite(relations["sign_agreement"][i, j]) else None,
                "sign_relation": sign_relation,
            }


def sort_pairs(pairs: List[dict], key: str, reverse: bool = True, top_k: int = 10) -> List[dict]:
    usable = [p for p in pairs if p.get(key) is not None and math.isfinite(float(p[key]))]
    usable.sort(key=lambda p: float(p[key]), reverse=reverse)
    return usable[:top_k]


def top_overlap(pairs: List[dict], top_k: int) -> dict:
    by_pearson = sort_pairs([{**p, "score": abs(p["pearson"])} for p in pairs if p["pearson"] is not None], "score", True, top_k)
    by_spearman = sort_pairs([{**p, "score": abs(p["spearman"])} for p in pairs if p["spearman"] is not None], "score", True, top_k)
    set_p = {(p["i"], p["j"]) for p in by_pearson}
    set_s = {(p["i"], p["j"]) for p in by_spearman}
    common = sorted(set_p & set_s)
    return {
        "top_k": top_k,
        "pearson_top_pairs": [[int(i), int(j)] for i, j in sorted(set_p)],
        "spearman_top_pairs": [[int(i), int(j)] for i, j in sorted(set_s)],
        "common_pairs": [[int(i), int(j)] for i, j in common],
        "overlap_count": int(len(common)),
        "overlap_fraction": float(len(common) / top_k) if top_k > 0 else None,
    }


def build_report(
    pearson: np.ndarray,
    spearman: np.ndarray,
    relations: dict,
    pairs: List[dict],
    include_diagonal: bool,
    eps: float,
    top_k: int,
    pearson_path: Path,
    spearman_path: Path,
) -> dict:
    n = pearson.shape[0]
    mask = offdiag_mask(n, include_diagonal=include_diagonal)
    p_vec = pearson[mask]
    s_vec = spearman[mask]
    finite_pair_mask = np.isfinite(p_vec) & np.isfinite(s_vec)
    sign_agreement_values = relations["sign_agreement"][mask]

    return {
        "input": {
            "pearson_file": str(pearson_path),
            "spearman_file": str(spearman_path),
            "shape": list(pearson.shape),
            "include_diagonal": include_diagonal,
            "eps": eps,
        },
        "summary": {
            "finite_pair_values": int(np.sum(finite_pair_mask)),
            "matrix_vector_corr": vector_corr(p_vec, s_vec),
            "matrix_cosine_similarity": cosine_similarity(p_vec, s_vec),
            "sign_agreement_fraction": fraction(sign_agreement_values),
            "absdiff": matrix_basic_stats(relations["absdiff"], mask),
            "signed_diff": matrix_basic_stats(relations["signed_diff"], mask),
            "abs_ratio": matrix_basic_stats(relations["abs_ratio"], mask),
            "signed_ratio": matrix_basic_stats(relations["signed_ratio"], mask),
            "mean_abs_strength": matrix_basic_stats(relations["mean_abs_strength"], mask),
            "abs_dominance": matrix_basic_stats(relations["abs_dominance"], mask),
        },
        "top": {
            "largest_absdiff": sort_pairs(pairs, "absdiff", True, top_k),
            "largest_mean_abs_strength": sort_pairs(pairs, "mean_abs_strength", True, top_k),
            "largest_abs_ratio": sort_pairs(pairs, "abs_ratio", True, top_k),
            "pearson_stronger_than_spearman": sort_pairs(pairs, "abs_dominance", True, top_k),
            "spearman_stronger_than_pearson": sort_pairs(pairs, "abs_dominance", False, top_k),
            "top_overlap": top_overlap(pairs, top_k),
        },
    }


def write_pairs_csv(path: Path, pairs: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "i",
        "j",
        "label_i",
        "label_j",
        "pearson",
        "spearman",
        "signed_diff",
        "absdiff",
        "signed_ratio",
        "abs_ratio",
        "mean_abs_strength",
        "abs_dominance",
        "sign_agreement",
        "sign_relation",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def save_outputs(
    out_prefix: Path,
    relations: dict,
    report: dict,
    pairs: List[dict],
    labels: List[str],
    save_csv: bool,
    save_png: bool,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for name, matrix in relations.items():
        np.save(out_prefix.with_name(out_prefix.name + f"_{name}.npy"), matrix)

    report_path = out_prefix.with_name(out_prefix.name + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pairs_path = out_prefix.with_name(out_prefix.name + "_pairs.csv")
    write_pairs_csv(pairs_path, pairs)

    if save_csv:
        for name, matrix in relations.items():
            save_matrix_csv(out_prefix.with_name(out_prefix.name + f"_{name}.csv"), matrix, labels)

    if save_png:
        # Для ratio не задаём vmin/vmax отдельно, поэтому используем общий helper только для матриц в диапазоне около [-1;1].
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_signed_diff.png"), relations["signed_diff"], labels, "Pearson - Spearman")
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_absdiff.png"), relations["absdiff"], labels, "|Pearson - Spearman|")
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_mean_abs_strength.png"), relations["mean_abs_strength"], labels, "Средняя сила связи")
        save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_sign_agreement.png"), relations["sign_agreement"], labels, "Совпадение знака")

    print(f"Сохранён отчёт          : {report_path}")
    print(f"Сохранена таблица пар   : {pairs_path}")
    print(f"Сохранены матрицы .npy  : {out_prefix.parent / (out_prefix.name + '_*.npy')}")
    if save_csv:
        print("Дополнительно сохранены CSV-матрицы.")
    if save_png:
        print("Дополнительно сохранены PNG-теплокарты.")


def print_summary(report: dict, pairs: List[dict], top_k: int) -> None:
    summary = report["summary"]
    print("=== СВОДКА ВЗАИМНЫХ СООТНОШЕНИЙ ===")
    print(f"корреляция двух матриц как векторов : {summary['matrix_vector_corr']}")
    print(f"косинусное сходство матриц          : {summary['matrix_cosine_similarity']}")
    print(f"доля совпадения знаков              : {summary['sign_agreement_fraction']}")
    print(f"средний |P-S|                       : {summary['absdiff']['mean']}")
    print(f"максимальный |P-S|                  : {summary['absdiff']['max']}")
    print(f"среднее |P|/|S|                     : {summary['abs_ratio']['mean']}")
    print()

    print(f"=== ТОП-{top_k}: где Пирсон и Спирмен сильнее всего различаются ===")
    for p in report["top"]["largest_absdiff"]:
        print(
            f"{p['label_i']} - {p['label_j']}: "
            f"P={p['pearson']:.6f}, S={p['spearman']:.6f}, |P-S|={p['absdiff']:.6f}, {p['sign_relation']}"
        )
    print()

    print(f"=== ТОП-{top_k}: самые сильные связи по средней силе |P| и |S| ===")
    for p in report["top"]["largest_mean_abs_strength"]:
        print(
            f"{p['label_i']} - {p['label_j']}: "
            f"P={p['pearson']:.6f}, S={p['spearman']:.6f}, mean_abs={p['mean_abs_strength']:.6f}"
        )
    print()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Строит матрицы и таблицы взаимных соотношений между корреляциями Пирсона и Спирмена."
    )
    parser.add_argument("--prefix", required=True, help="Префикс входных файлов: <prefix>_pearson.npy и <prefix>_spearman.npy")
    parser.add_argument("--out", default="data/sessions/correlation_relations", help="Префикс выходных файлов без расширения")
    parser.add_argument("--meta-json", default="", help="JSON-метаданные для имён каналов")
    parser.add_argument("--eps", type=float, default=EPS_DEFAULT, help="Порог около нуля для деления и знаков")
    parser.add_argument("--top-k", type=int, default=10, help="Размер топов в отчёте")
    parser.add_argument("--include-diagonal", action="store_true", help="Учитывать диагональ в сводной статистике")
    parser.add_argument("--save-csv", action="store_true", help="Сохранить не только таблицу пар, но и все матрицы в CSV")
    parser.add_argument("--save-png", action="store_true", help="Сохранить основные матрицы как PNG-теплокарты")
    parser.add_argument("--print-summary", action="store_true", help="Напечатать краткую сводку и топы")
    parser.add_argument("--print-matrix", choices=list(build_relation_matrices(np.eye(1), np.eye(1), EPS_DEFAULT).keys()), help="Напечатать одну из итоговых матриц")
    parser.add_argument("--matrix-decimals", type=int, default=3)
    parser.add_argument("--matrix-max-rows", type=int, default=20)
    parser.add_argument("--matrix-max-cols", type=int, default=20)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        prefix = Path(args.prefix)
        out_prefix = Path(args.out)

        pearson, spearman, pearson_path, spearman_path = load_pair_from_prefix(prefix)
        validate_same_shape(pearson, spearman)

        labels = load_channel_names(args.meta_json or None, pearson.shape[0])
        if not labels or len(labels) != pearson.shape[0]:
            labels = default_channel_names(pearson.shape[0])

        relations = build_relation_matrices(pearson, spearman, args.eps)
        pairs = list(iter_pairs(pearson, spearman, relations, labels, args.eps))
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

        save_outputs(out_prefix, relations, report, pairs, labels, args.save_csv, args.save_png)

        if args.print_summary:
            print_summary(report, pairs, args.top_k)

        if args.print_matrix:
            print(f"=== МАТРИЦА {args.print_matrix} ===")
            print(
                format_matrix_text(
                    relations[args.print_matrix],
                    labels,
                    decimals=args.matrix_decimals,
                    max_rows=args.matrix_max_rows,
                    max_cols=args.matrix_max_cols,
                )
            )

        return 0
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка в correlation_relations.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
