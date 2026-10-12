"""AllGather 独立入口：默认图形界面，--cli 输出逐轮状态。"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AllGather.simulator import DEMO_INPUTS, all_gather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", action="store_true", help="只在终端打印各轮状态")
    args = parser.parse_args()
    if args.cli:
        all_gather(DEMO_INPUTS)
    else:
        import tkinter as tk
        from AllGather.gui import AllGatherWindow

        root = tk.Tk()
        AllGatherWindow(root)
        root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
