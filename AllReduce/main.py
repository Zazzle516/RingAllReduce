"""GUI by default; --cli exposes the same event scheduler in a terminal."""
import argparse
import json
from pathlib import Path
import sys

# 支持从任意工作目录直接运行此文件，也支持 python -m AllReduce
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AllReduce.simulator import RingAllReduceSimulator, demo_inputs


def format_event(state):
    e = state['event']
    before, after = e['before'], e['after']
    text = (f"#{e['id']:04} tick={e['tick']:03} C{e['channel']} "
            f"{chr(65+e['source'])}->{chr(65+e['target'])} "
            f"R{e['round']} chunk={e['chunk']} slice={e['part']} {e['kind']} "
            f"H/T {before['head']}/{before['tail']} -> {after['head']}/{after['tail']}")
    if 'payload' in e:
        text += f" values={e['payload']['values']}"
    if 'result' in e:
        text += f" result={e['result']['values']}"
    return text


def main():
    parser = argparse.ArgumentParser(description="Four-GPU multi-channel SimpleBuffer simulator")
    parser.add_argument('--cli', action='store_true')
    parser.add_argument('--step', action='store_true', help='CLI: Enter advances one event')
    parser.add_argument('--rounds', action='store_true', help='CLI: advance to next common round milestone')
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--buffer-slots', type=int, default=2)
    parser.add_argument('--slice-elements', type=int, default=1)
    parser.add_argument('--slow-rank', type=int, default=3, choices=range(4))
    parser.add_argument('--receive-period', type=int, default=4,
                        help='Slow GPU consumes once every N scheduler sweeps; 1 disables delay')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--quiet', action='store_true', help='CLI: show final summary only')
    args = parser.parse_args()
    if not args.cli and any((args.step, args.rounds, args.export, args.quiet)):
        parser.error('--step/--rounds/--export/--quiet require --cli')
    try:
        data = json.loads(args.input.read_text(encoding='utf-8-sig')) if args.input else demo_inputs()
        simulator = RingAllReduceSimulator(data, nchannels=args.channels, buffer_slots=args.buffer_slots,
                           slice_elements=args.slice_elements, slow_rank=args.slow_rank,
                           receive_period=args.receive_period,
                           record_history=not args.cli or bool(args.export))
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.cli:
        print(f"CPU simulation | {simulator.nchannels} channels | slots={simulator.buffer_slots} | "
              f"slice elements={simulator.slice_elements} | input/padded={simulator.size}/{simulator.padded_size}")
        for c in simulator.channels:
            print(f"Channel {c.id}: [{c.start}, {c.stop}), chunk={c.chunk_size}, slices/chunk={c.parts}")
        while not simulator.done:
            if args.step:
                input('Enter: next ' + ('round milestone' if args.rounds else 'event') + '...')
            state = simulator.step_round() if args.rounds else simulator.step_event()
            if not args.quiet:
                print(format_event(state))
                if args.rounds:
                    print(f"All channels have completed at least round {simulator.round}")
        ok = simulator.verify()
        print('Verification: ' + ('PASS' if ok else 'FAIL'))
        print('Result:', simulator.results[0])
        print('Events:', simulator.event_id, simulator.counts)
        print('Sent elements per GPU:', {r: sum(simulator.gpus[r].channels[c.id].send_link.elements_sent for c in simulator.channels) for r in simulator.ring})
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(simulator.trace(), ensure_ascii=False, indent=2), encoding='utf-8')
            print('Trace:', args.export.resolve())
        return 0 if ok else 1
    try:
        from AllReduce.gui import launch
    except ImportError:
        print('Tkinter unavailable. Use python -m AllReduce --cli', file=sys.stderr)
        return 1
    launch(simulator)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
