"""单 channel Ring AllGather 教学模拟：只复制和转发，不做归约。

运行：python all-gather-sim.py
为便于观察，这里按同步轮次推进，不模拟真实 GPU 或异步流水线。
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class GPU:
    rank: int
    # 下标是数据来源 rank；None 表示尚未收到，不能用零冒充有效数据。
    blocks: list[tuple[int, ...] | None]


def all_gather(
    inputs: Sequence[Sequence[int]],
    ring: Sequence[int] = (0, 3, 2, 1),
    verbose: bool = True,
) -> list[list[int]]:
    """返回按接收 rank 排列的结果，每份结果内部按来源 rank 拼接。

    ring 只决定传输路径，不改变输出的数据排列顺序。
    每张 GPU 的输入长度必须相同；每轮沿环发送一个完整数据块。
    """
    nranks = len(inputs)
    if nranks == 0 or len(ring) != nranks or set(ring) != set(range(nranks)):
        raise ValueError("ring 必须包含每个输入 rank，且恰好出现一次")
    if any(len(row) != len(inputs[0]) for row in inputs):
        raise ValueError("每张 GPU 的输入长度必须相同")

    gpus = [GPU(rank, [None] * nranks) for rank in range(nranks)]
    for gpu, values in zip(gpus, inputs):
        gpu.blocks[gpu.rank] = tuple(values)

    def show_state():
        for gpu in gpus:
            blocks = [list(block) if block is not None else None for block in gpu.blocks]
            print(f"  GPU {gpu.rank}: {blocks}")

    if verbose:
        print("环路径：" + " -> ".join(map(str, (*ring, ring[0]))))
        print("初始状态（槽位按来源 rank 排列，None 表示未收到）：")
        show_state()

    for step in range(nranks - 1):
        if verbose:
            print(f"\n第 {step + 1} 轮：")

        # 先收集本轮所有发送，再统一接收，避免同一块在一轮内跨越多跳。
        transfers = []
        for position, source in enumerate(ring):
            target = ring[(position + 1) % nranks]
            # 第 1 轮发送自己的块，此后发送上一轮从上游收到的块。
            owner = ring[(position - step) % nranks]
            block = gpus[source].blocks[owner]
            assert block is not None, "发送前必须已经拥有该块"
            transfers.append((target, owner, block))
            if verbose:
                print(f"  GPU {source} -> GPU {target}: 来源 GPU {owner}，数据 {list(block)}")

        for target, owner, block in transfers:
            assert gpus[target].blocks[owner] is None, "每个来源的块只应接收一次"
            gpus[target].blocks[owner] = block  # 不可变 tuple；逻辑上复制，无加法。

        if verbose:
            show_state()

    results = []
    for gpu in gpus:
        row = []
        for block in gpu.blocks:
            assert block is not None, "结束时必须拥有全部来源的数据块"
            row.extend(block)
        results.append(row)

    expected = [value for values in inputs for value in values]
    assert all(row == expected for row in results)
    if verbose:
        print("\n最终结果（按来源 rank 拼接）：")
        for rank, row in enumerate(results):
            print(f"  GPU {rank}: {row}")
        print(f"验证通过：{nranks - 1} 轮后，每张 GPU 都拥有全部输入。")
    return results


if __name__ == "__main__":
    all_gather([
        [10, 11],  # GPU 0
        [20, 21],  # GPU 1
        [30, 31],  # GPU 2
        [40, 41],  # GPU 3
    ])
