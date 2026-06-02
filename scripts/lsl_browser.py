#!/usr/bin/env python3
import argparse
import sys
from typing import List

from pylsl import resolve_streams


def format_rate(rate: float) -> str:
    if rate == 0:
        return "нерегулярная"
    return f"{rate:g} Hz"


def safe_call(fn, default=""):
    try:
        return fn()
    except Exception:
        return default


def discover_streams(timeout: float) -> List:
    return resolve_streams(wait_time=timeout)


def print_streams(streams) -> None:
    if not streams:
        print("LSL-потоки не найдены.")
        return

    print(f"Найдено LSL-потоков: {len(streams)}\n")
    for i, info in enumerate(streams):
        name = safe_call(info.name, "")
        type_ = safe_call(info.type, "")
        channels = safe_call(info.channel_count, "")
        srate = safe_call(info.nominal_srate, 0)
        source_id = safe_call(info.source_id, "")
        uid = safe_call(info.uid, "")
        hostname = safe_call(info.hostname, "")

        print(f"[{i}]")
        print(f"  имя          : {name}")
        print(f"  тип          : {type_}")
        print(f"  каналы       : {channels}")
        print(f"  частота      : {format_rate(srate)}")
        print(f"  источник     : {source_id}")
        print(f"  uid          : {uid}")
        print(f"  хост         : {hostname}")
        print()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ищет доступные LSL-потоки и выводит краткую сводку."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Таймаут поиска в секундах (по умолчанию: 2.0)",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        streams = discover_streams(args.timeout)
        print_streams(streams)
        return 0
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:
        print(f"Ошибка при поиске потоков: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
