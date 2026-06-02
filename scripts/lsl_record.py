#!/usr/bin/env python3
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import numpy as np
from pylsl import StreamInlet, resolve_streams


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


def record_loop(inlet: StreamInlet, duration: float, pull_timeout: float):
    samples = []
    timestamps = []

    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        sample, ts = inlet.pull_sample(timeout=pull_timeout)
        if sample is None:
            continue
        samples.append(sample)
        timestamps.append(ts)

    if not samples:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    samples_array = np.asarray(samples, dtype=float)
    timestamps_array = np.asarray(timestamps, dtype=float)
    return samples_array, timestamps_array


def compute_summary(samples: np.ndarray, timestamps: np.ndarray, nominal_srate: float) -> dict:
    n_samples = int(samples.shape[0]) if samples.ndim == 2 else 0
    n_channels = int(samples.shape[1]) if samples.ndim == 2 else 0

    summary = {
        "n_samples": n_samples,
        "n_channels": n_channels,
        "nominal_srate": float(nominal_srate),
        "expected_samples": None,
        "actual_duration_seconds": None,
        "mean_dt": None,
        "min_dt": None,
        "max_dt": None,
    }

    if nominal_srate and nominal_srate > 0 and timestamps.size > 1:
        actual_duration = float(timestamps[-1] - timestamps[0])
        expected_samples = int(round(actual_duration * nominal_srate))
        summary["expected_samples"] = expected_samples
        summary["actual_duration_seconds"] = actual_duration

    if timestamps.size > 1:
        dts = np.diff(timestamps)
        summary["mean_dt"] = float(np.mean(dts))
        summary["min_dt"] = float(np.min(dts))
        summary["max_dt"] = float(np.max(dts))

    return summary


def save_outputs(
    samples: np.ndarray,
    timestamps: np.ndarray,
    info: dict,
    summary: dict,
    out_prefix: Path,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    samples_path = out_prefix.with_name(out_prefix.name + "_samples.npy")
    timestamps_path = out_prefix.with_name(out_prefix.name + "_timestamps.npy")
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")
    info_path = out_prefix.with_name(out_prefix.name + "_info.txt")

    np.save(samples_path, samples)
    np.save(timestamps_path, timestamps)

    meta_payload = {
        "stream": {k: v for k, v in info.items() if k != "xml"},
        "summary": summary,
    }
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    info_lines = [
        f"имя: {info.get('name', '')}",
        f"тип: {info.get('type', '')}",
        f"число_каналов: {info.get('channel_count', '')}",
        f"частота: {info.get('nominal_srate', '')}",
        f"источник: {info.get('source_id', '')}",
        f"uid: {info.get('uid', '')}",
        f"сессия: {info.get('session_id', '')}",
        f"хост: {info.get('hostname', '')}",
        f"число_сэмплов: {summary.get('n_samples')}",
        f"число_каналов_факт: {summary.get('n_channels')}",
        f"ожидаемое_число_сэмплов: {summary.get('expected_samples')}",
        f"фактическая_длительность_сек: {summary.get('actual_duration_seconds')}",
        f"средний_dt: {summary.get('mean_dt')}",
        f"минимальный_dt: {summary.get('min_dt')}",
        f"максимальный_dt: {summary.get('max_dt')}",
    ]
    info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    print(f"Сохранены сэмплы         : {samples_path}")
    print(f"Сохранены метки времени  : {timestamps_path}")
    print(f"Сохранены метаданные     : {meta_path}")
    print(f"Сохранена сводка         : {info_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Записывает данные из LSL-потока в файлы .npy."
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Таймаут поиска потоков в секундах.")
    parser.add_argument("--index", type=int, default=None, help="Индекс в списке отфильтрованных потоков.")
    parser.add_argument("--name", type=str, default=None, help="Точное имя потока.")
    parser.add_argument("--type", dest="type_", type=str, default="EEG", help="Точный тип потока (по умолчанию: EEG).")
    parser.add_argument("--duration", type=float, default=60.0, help="Длительность записи в секундах.")
    parser.add_argument(
        "--pull-timeout",
        type=float,
        default=1.0,
        help="Таймаут для pull_sample в секундах (по умолчанию: 1.0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/sessions/session_001",
        help="Префикс выходного пути без расширения.",
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

        print("Выбранный поток:")
        print(f"  имя          : {meta['name']}")
        print(f"  тип          : {meta['type']}")
        print(f"  каналы       : {meta['channel_count']}")
        print(f"  частота      : {format_rate(meta['nominal_srate'])}")
        print()

        inlet = StreamInlet(info)

        print(f"Идёт запись в течение {args.duration:g} сек...")
        samples, timestamps = record_loop(inlet, args.duration, args.pull_timeout)

        if samples.size == 0:
            print("Сэмплы не получены.", file=sys.stderr)
            return 3

        summary = compute_summary(samples, timestamps, meta["nominal_srate"])
        save_outputs(samples, timestamps, meta, summary, Path(args.out))

        print("\n=== СВОДКА ===")
        print(f"Получено сэмплов           : {summary['n_samples']}")
        print(f"Число каналов             : {summary['n_channels']}")
        print(f"Частота                   : {summary['nominal_srate']}")
        print(f"Ожидаемое число сэмплов   : {summary['expected_samples']}")
        print(f"Фактическая длительность  : {summary['actual_duration_seconds']}")
        print(f"Средний dt                : {summary['mean_dt']}")
        print(f"Минимальный dt            : {summary['min_dt']}")
        print(f"Максимальный dt           : {summary['max_dt']}")

        return 0
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка при записи потока: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
