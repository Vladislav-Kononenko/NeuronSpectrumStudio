#!/usr/bin/env python3
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from pylsl import resolve_streams


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
        print("Найдено несколько подходящих потоков. Используется первый.\n")
        for i, s in enumerate(filtered):
            print(
                f"[{i}] имя={safe_call(s.name, '')} "
                f"тип={safe_call(s.type, '')} "
                f"каналы={safe_call(s.channel_count, '')} "
                f"частота={format_rate(safe_call(s.nominal_srate, 0))}"
            )
        print()

    return filtered[0]


def parse_xml(xml_text: str):
    return ET.fromstring(xml_text)


def extract_channel_names_from_xml(xml_text: str) -> List[str]:
    names = []
    try:
        root = parse_xml(xml_text)
    except ET.ParseError:
        return names

    for ch in root.findall(".//desc/channels/channel"):
        label = ch.findtext("label")
        if label:
            names.append(label.strip())

    return names


def extract_metadata_dict(xml_text: str) -> dict:
    root = parse_xml(xml_text)

    meta = {
        "name": root.findtext("name"),
        "type": root.findtext("type"),
        "channel_count": int(root.findtext("channel_count", "0")),
        "nominal_srate": float(root.findtext("nominal_srate", "0")),
        "channel_format": root.findtext("channel_format"),
        "source_id": root.findtext("source_id"),
        "uid": root.findtext("uid"),
        "session_id": root.findtext("session_id"),
        "hostname": root.findtext("hostname"),
        "channels": [],
    }

    for ch in root.findall(".//desc/channels/channel"):
        meta["channels"].append(
            {
                "label": ch.findtext("label"),
                "unit": ch.findtext("unit"),
                "type": ch.findtext("type"),
            }
        )

    return meta


def print_basic_info(info, xml_text: str) -> None:
    print("=== ОСНОВНАЯ ИНФОРМАЦИЯ ===")
    print(f"имя          : {safe_call(info.name, '')}")
    print(f"тип          : {safe_call(info.type, '')}")
    print(f"каналы       : {safe_call(info.channel_count, '')}")
    print(f"частота      : {format_rate(safe_call(info.nominal_srate, 0))}")
    print(f"источник     : {safe_call(info.source_id, '')}")
    print(f"uid          : {safe_call(info.uid, '')}")
    print(f"сессия       : {safe_call(info.session_id, '')}")
    print(f"хост         : {safe_call(info.hostname, '')}")
    print()

    channel_names = extract_channel_names_from_xml(xml_text)
    if channel_names:
        print("=== ИМЕНА КАНАЛОВ ===")
        for i, name in enumerate(channel_names, start=1):
            print(f"{i:>2}. {name}")
        print()
    else:
        print("Имена каналов не найдены в XML-метаданных.\n")


def save_outputs(xml_text: str, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    xml_path = out_prefix.with_suffix(".xml")
    json_path = out_prefix.with_suffix(".json")

    xml_path.write_text(xml_text, encoding="utf-8")
    metadata = extract_metadata_dict(xml_text)
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Сохранены XML-метаданные  : {xml_path}")
    print(f"Сохранены JSON-метаданные : {json_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Показывает данные выбранного LSL-потока и сохраняет его метаданные."
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Таймаут поиска потоков в секундах.")
    parser.add_argument("--index", type=int, default=None, help="Индекс в списке отфильтрованных потоков.")
    parser.add_argument("--name", type=str, default=None, help="Точное имя потока.")
    parser.add_argument("--type", dest="type_", type=str, default=None, help="Точный тип потока.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/sessions/stream_meta",
        help="Префикс выходного пути без расширения (по умолчанию: data/sessions/stream_meta)",
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
        xml_text = info.as_xml()

        print_basic_info(info, xml_text)
        save_outputs(xml_text, Path(args.out))
        return 0
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка при анализе потока: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
