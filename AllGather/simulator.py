"""单 channel Ring AllGather 教学模拟：只复制和转发，不做归约。

图形界面：python AllGather/main.py
终端输出：python AllGather/main.py --cli
为便于观察，这里按同步轮次推进，不模拟真实 GPU 或异步流水线。
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class GPU:
    rank: int
    # 下标是数据来源 rank；None 表示尚未收到，不能用零冒充有效数据。
    blocks: list[tuple[int, ...] | None]


def gather_rounds(
    inputs: Sequence[Sequence[int]],
    ring: Sequence[int] = (0, 3, 2, 1),
):
    """逐轮生成独立快照，GUI 和终端共用同一个算法。

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

    def snapshot(round_id, transfers):
        return {"round": round_id, "transfers": tuple(transfers),
                "blocks": tuple(tuple(gpu.blocks) for gpu in gpus)}

    yield snapshot(0, [])

    for step in range(nranks - 1):
        # 先收集本轮所有发送，再统一接收，避免同一块在一轮内跨越多跳。
        transfers = []
        for position, source in enumerate(ring):
            target = ring[(position + 1) % nranks]
            # 第 1 轮发送自己的块，此后发送上一轮从上游收到的块。
            owner = ring[(position - step) % nranks]
            block = gpus[source].blocks[owner]
            assert block is not None, "发送前必须已经拥有该块"
            transfers.append((source, target, owner, block))

        for source, target, owner, block in transfers:
            assert gpus[target].blocks[owner] is None, "每个来源的块只应接收一次"
            gpus[target].blocks[owner] = block  # 不可变 tuple；逻辑上复制，无加法。

        yield snapshot(step + 1, transfers)


def all_gather(
    inputs: Sequence[Sequence[int]],
    ring: Sequence[int] = (0, 3, 2, 1),
    verbose: bool = True,
) -> list[list[int]]:
    """运行全部轮次，保留终端/API 用法；输出按来源 rank 拼接。"""
    for state in gather_rounds(inputs, ring):
        if verbose:
            print(f"\n第 {state['round']} 轮（0 为初始状态）：")
            for source, target, owner, block in state["transfers"]:
                print(f"  GPU {source} -> GPU {target}: 来源 GPU {owner}，数据 {list(block)}")
            for rank, blocks in enumerate(state["blocks"]):
                print(f"  GPU {rank}: {[list(b) if b is not None else None for b in blocks]}")

    results = []
    for blocks in state["blocks"]:
        row = []
        for block in blocks:
            assert block is not None, "结束时必须拥有全部来源的数据块"
            row.extend(block)
        results.append(row)

    expected = [value for values in inputs for value in values]
    assert all(row == expected for row in results)
    if verbose:
        print("\n最终结果（按来源 rank 拼接）：")
        for rank, row in enumerate(results):
            print(f"  GPU {rank}: {row}")
        print(f"验证通过：{len(inputs) - 1} 轮后，每张 GPU 都拥有全部输入。")
    return results


DEMO_INPUTS = ((10, 11), (20, 21), (30, 31), (40, 41))
DEMO_RING = (0, 3, 2, 1)
