"""Run with python main.py (GUI), or python main.py --cli."""
import argparse
import json
from pathlib import Path
import sys

from simulator import GPUContainer, demo_inputs


def print_round(state: dict) -> None:
    print(f"\n=== Round {state['round']} / 6 : {state['phase']} ===")
    for event in state["transfers"]:
        src, dst = chr(65 + event['source']), chr(65 + event['target'])
        print(f"  {src} -> {dst}  chunk {event['chunk']}  "
              f"{event['payload']['values']} -> {event['result']['values']}")
        print(f"    {event['operation']} | {event['route']} | slot {event['slot']}")
        print(f"    head/tail: {event['before']} -> {event['published']} -> {event['consumed']}")
    for gpu in state['gpus'].values():
        print(f"  GPU {gpu['name']}: " + " | ".join(str(c['values']) for c in gpu['chunks']))


def main() -> int:
    parser = argparse.ArgumentParser(description="Four-GPU PCIe P2P RingAllReduce CPU simulator")
    parser.add_argument("--cli", action="store_true", help="Print all rounds in the terminal")
    parser.add_argument("--step", action="store_true", help="CLI: press Enter between rounds")
    parser.add_argument("--input", type=Path, help="JSON file containing four equal-length arrays")
    parser.add_argument("--export", type=Path, help="CLI: write full JSON trace to this file")
    args = parser.parse_args()
    if args.step and not args.cli:
        parser.error("--step requires --cli")
    if args.export and not args.cli:
        parser.error("--export requires --cli; the GUI has its own export button")
    try:
        inputs = json.loads(args.input.read_text(encoding="utf-8-sig")) if args.input else demo_inputs()
        container = GPUContainer(inputs)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.cli:
        print("CPU-only simulation | assumed PCIe P2P | A -> D -> C -> B -> A")
        print(f"Elements per GPU: {container.size}; padded: {container.padded_size}")
        print_round(container.history[0])
        while not container.done:
            if args.step:
                input("Press Enter for the next round...")
            print_round(container.step())
        print(f"\nExpected: {container.expected}")
        ok = container.verify()
        print("Verification: " + ("PASS" if ok else "FAIL"))
        print(f"Sent per GPU: {6 * container.chunk_size} simulated elements")
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(container.trace(), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Trace: {args.export.resolve()}")
        return 0 if ok else 1
    try:
        from gui import launch
        launch(container)
    except ImportError:
        print("Tkinter is unavailable. Use: python main.py --cli", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
