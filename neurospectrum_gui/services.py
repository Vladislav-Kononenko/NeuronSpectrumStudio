from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts import check_recording, compare_matrices, correlation_relations, lsl_inspect, lsl_record, matrix_tools


@dataclass(slots=True)
class StreamDescriptor:
    index: int
    name: str
    type: str
    source_id: str
    nominal_srate: float
    channel_count: int
    channel_format: str
    uid: str
    hostname: str


def safe_channel_labels(labels: list[str] | None, n_channels: int) -> list[str]:
    if labels and len(labels) == n_channels:
        return labels
    return matrix_tools.default_channel_names(n_channels)


def discover_lsl_streams(timeout: float = 2.0) -> list[StreamDescriptor]:
    streams = lsl_record.discover_streams(timeout)
    result: list[StreamDescriptor] = []
    for index, info in enumerate(streams):
        result.append(
            StreamDescriptor(
                index=index,
                name=lsl_record.safe_call(info.name, ""),
                type=lsl_record.safe_call(info.type, ""),
                source_id=lsl_record.safe_call(info.source_id, ""),
                nominal_srate=float(lsl_record.safe_call(info.nominal_srate, 0.0) or 0.0),
                channel_count=int(lsl_record.safe_call(info.channel_count, 0) or 0),
                channel_format=lsl_record.safe_call(info.channel_format, ""),
                uid=lsl_record.safe_call(info.uid, ""),
                hostname=lsl_record.safe_call(info.hostname, ""),
            )
        )
    return result


def resolve_stream(descriptor: StreamDescriptor, timeout: float = 2.0):
    streams = lsl_record.discover_streams(timeout)
    if not streams:
        raise RuntimeError("LSL-потоки не найдены.")

    for info in streams:
        if (
            lsl_record.safe_call(info.name, "") == descriptor.name
            and lsl_record.safe_call(info.type, "") == descriptor.type
            and lsl_record.safe_call(info.source_id, "") == descriptor.source_id
        ):
            return info

    if descriptor.index < len(streams):
        return streams[descriptor.index]
    raise RuntimeError("Не удалось повторно найти выбранный LSL-поток.")


def inspect_lsl_stream(descriptor: StreamDescriptor, timeout: float = 2.0) -> dict[str, Any]:
    info = resolve_stream(descriptor, timeout)
    xml_text = info.as_xml()
    metadata = lsl_inspect.extract_metadata_dict(xml_text)
    channel_names = lsl_inspect.extract_channel_names_from_xml(xml_text)
    summary = {
        "name": descriptor.name,
        "type": descriptor.type,
        "source_id": descriptor.source_id,
        "nominal_srate": descriptor.nominal_srate,
        "channel_count": descriptor.channel_count,
        "channel_format": descriptor.channel_format,
        "uid": descriptor.uid,
        "hostname": descriptor.hostname,
    }
    return {
        "summary": summary,
        "xml_text": xml_text,
        "metadata": metadata,
        "channel_names": channel_names,
    }


def save_stream_metadata(descriptor: StreamDescriptor, out_prefix: str | Path, timeout: float = 2.0) -> dict[str, str]:
    inspected = inspect_lsl_stream(descriptor, timeout)
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    xml_path = prefix.with_suffix(".xml")
    json_path = prefix.with_suffix(".json")
    xml_path.write_text(inspected["xml_text"], encoding="utf-8")
    json_path.write_text(json.dumps(inspected["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"xml": str(xml_path), "json": str(json_path)}


def analyze_recording(
    samples_path: str | Path,
    timestamps_path: str | Path,
    sfreq: float,
    window_seconds: float,
    start_seconds: float,
    out_prefix: str | Path | None = None,
    meta_json: str | Path | None = None,
    save_csv: bool = False,
    save_png: bool = False,
    save_relations: bool = True,
    top_k: int = 10,
) -> dict[str, Any]:
    samples_path = Path(samples_path)
    timestamps_path = Path(timestamps_path)

    samples = check_recording.load_array(samples_path, "сэмплы")
    timestamps = check_recording.load_array(timestamps_path, "временные метки")

    orientation = check_recording.detect_orientation(samples, timestamps)
    data_cs = check_recording.to_channels_samples(samples, orientation)
    window, ts_window, start_idx, end_idx = check_recording.select_window_by_samples(
        data_cs=data_cs,
        timestamps=timestamps,
        sfreq=sfreq,
        window_seconds=window_seconds,
        start_seconds=start_seconds,
    )

    pearson = check_recording.corrcoef_safe(window)
    spearman = check_recording.corrcoef_safe(check_recording.spearman_rank_rows(window))
    diff_abs = np.abs(pearson - spearman)

    n_channels = int(data_cs.shape[0])
    labels = matrix_tools.load_channel_names(str(meta_json) if meta_json else None, n_channels)
    labels = safe_channel_labels(labels, n_channels)

    ts_summary = check_recording.summarize_timestamps(timestamps)
    pearson_stats = check_recording.matrix_stats(pearson, "Пирсон")
    spearman_stats = check_recording.matrix_stats(spearman, "Спирмен")
    compare_stats = matrix_tools.compare_matrices(pearson, spearman)

    report = {
        "recording": {
            "samples_file": str(samples_path),
            "timestamps_file": str(timestamps_path),
            "raw_samples_shape": list(samples.shape),
            "detected_orientation": orientation,
            "data_channels_samples_shape": list(data_cs.shape),
            "sampling_rate_hz": sfreq,
        },
        "timestamps": ts_summary,
        "window": {
            "window_seconds": window_seconds,
            "start_seconds": start_seconds,
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
            {"i": i, "j": j, "label_i": labels[i], "label_j": labels[j], "value": value}
            for _, i, j, value in check_recording.top_pairs(pearson, top_k=top_k)
        ],
        "top_spearman_pairs": [
            {"i": i, "j": j, "label_i": labels[i], "label_j": labels[j], "value": value}
            for _, i, j, value in check_recording.top_pairs(spearman, top_k=top_k)
        ],
    }

    relations = correlation_relations.build_relation_matrices(
        pearson=pearson,
        spearman=spearman,
        eps=correlation_relations.EPS_DEFAULT,
    )
    relation_pairs = list(correlation_relations.iter_pairs(pearson, spearman, relations, labels, correlation_relations.EPS_DEFAULT))
    relation_report = correlation_relations.build_report(
        pearson=pearson,
        spearman=spearman,
        relations=relations,
        pairs=relation_pairs,
        include_diagonal=False,
        eps=correlation_relations.EPS_DEFAULT,
        top_k=top_k,
        pearson_path=Path(str(out_prefix) + "_pearson.npy") if out_prefix else Path("pearson.npy"),
        spearman_path=Path(str(out_prefix) + "_spearman.npy") if out_prefix else Path("spearman.npy"),
    )

    saved_paths: dict[str, str] = {}
    if out_prefix:
        out_prefix = Path(out_prefix)
        check_recording.save_outputs(
            out_prefix=out_prefix,
            pearson=pearson,
            spearman=spearman,
            diff_abs=diff_abs,
            report=report,
            labels=labels,
            save_csv=save_csv,
            save_png=save_png,
        )
        saved_paths["base_prefix"] = str(out_prefix)

        if save_relations:
            relation_prefix = out_prefix.with_name(out_prefix.name + "_relations")
            correlation_relations.save_outputs(
                out_prefix=relation_prefix,
                relations=relations,
                report=relation_report,
                pairs=relation_pairs,
                labels=labels,
                save_csv=save_csv,
                save_png=save_png,
            )
            saved_paths["relations_prefix"] = str(relation_prefix)

    return {
        "labels": labels,
        "pearson": pearson,
        "spearman": spearman,
        "diff_abs": diff_abs,
        "relations": relations,
        "report": report,
        "relation_report": relation_report,
        "saved_paths": saved_paths,
    }


def compare_result_sets(
    reference_prefix: str | Path,
    candidate_prefix: str | Path,
    meta_json: str | Path | None = None,
    out_prefix: str | Path | None = None,
    save_csv: bool = False,
    save_png: bool = False,
) -> dict[str, Any]:
    reference_prefix = Path(reference_prefix)
    candidate_prefix = Path(candidate_prefix)

    pearson_ref = compare_matrices.load_matrix(reference_prefix, "_pearson.npy")
    pearson_cand = compare_matrices.load_matrix(candidate_prefix, "_pearson.npy")
    spearman_ref = compare_matrices.load_matrix(reference_prefix, "_spearman.npy")
    spearman_cand = compare_matrices.load_matrix(candidate_prefix, "_spearman.npy")

    labels = matrix_tools.load_channel_names(str(meta_json) if meta_json else None, pearson_ref.shape[0])
    labels = safe_channel_labels(labels, pearson_ref.shape[0])

    pearson_diff_abs = np.abs(pearson_ref - pearson_cand)
    spearman_diff_abs = np.abs(spearman_ref - spearman_cand)
    pearson_diff_abs_display = matrix_tools.shift_matrix_rows_for_display(pearson_diff_abs)
    spearman_diff_abs_display = matrix_tools.shift_matrix_rows_for_display(spearman_diff_abs)

    pearson_stats = matrix_tools.compare_matrices(pearson_ref, pearson_cand)
    spearman_stats = matrix_tools.compare_matrices(spearman_ref, spearman_cand)

    report = {
        "reference_prefix": str(reference_prefix),
        "candidate_prefix": str(candidate_prefix),
        "pearson": pearson_stats,
        "spearman": spearman_stats,
    }

    if out_prefix:
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_prefix.with_name(out_prefix.name + "_pearson_absdiff.npy"), pearson_diff_abs_display)
        np.save(out_prefix.with_name(out_prefix.name + "_spearman_absdiff.npy"), spearman_diff_abs_display)
        out_prefix.with_name(out_prefix.name + "_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if save_csv:
            matrix_tools.save_matrix_csv(out_prefix.with_name(out_prefix.name + "_pearson_absdiff.csv"), pearson_diff_abs_display, labels)
            matrix_tools.save_matrix_csv(out_prefix.with_name(out_prefix.name + "_spearman_absdiff.csv"), spearman_diff_abs_display, labels)
        if save_png:
            matrix_tools.save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_pearson_absdiff.png"), pearson_diff_abs_display, labels, "|Пирсон: эталон - сравниваемый|")
            matrix_tools.save_matrix_heatmap(out_prefix.with_name(out_prefix.name + "_spearman_absdiff.png"), spearman_diff_abs_display, labels, "|Спирмен: эталон - сравниваемый|")

    return {
        "labels": labels,
        "pearson_absdiff": pearson_diff_abs_display,
        "spearman_absdiff": spearman_diff_abs_display,
        "report": report,
    }


def save_live_snapshot(
    out_prefix: str | Path,
    labels: list[str],
    pearson: np.ndarray,
    spearman: np.ndarray,
    diff_abs: np.ndarray,
    report: dict[str, Any],
    save_csv: bool,
    save_png: bool,
) -> str:
    out_prefix = Path(out_prefix)
    from scripts.live_correlations import save_latest

    save_latest(
        out_prefix=out_prefix,
        pearson=pearson,
        spearman=spearman,
        diff_abs=diff_abs,
        report=report,
        labels=labels,
        save_csv=save_csv,
        save_png=save_png,
    )
    return str(out_prefix)
