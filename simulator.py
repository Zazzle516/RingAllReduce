"""CPU-only, round-based PCIe P2P RingAllReduce teaching simulator.

No CUDA calls, device allocations, bandwidth prediction or real PCIe traffic.
Each round snapshots sends before any receive, modelling concurrent edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Sequence


@dataclass(frozen=True)
class Chunk:
    """A contiguous vector block and the ranks already included in its sum."""

    index: int
    values: tuple[Real, ...]
    contributors: frozenset[int]

    def reduce_with(self, other: Chunk) -> Chunk:
        if self.index != other.index or len(self.values) != len(other.values):
            raise ValueError("Only matching blocks can be reduced")
        if self.contributors & other.contributors:
            raise ValueError("A rank would be counted twice")
        return Chunk(self.index, tuple(a + b for a, b in zip(self.values, other.values)),
                     self.contributors | other.contributors)

    def to_dict(self) -> dict:
        return {"index": self.index, "values": list(self.values),
                "contributors": sorted(self.contributors)}


@dataclass
class GPU:
    """Logical device; input blocks stay immutable, working blocks evolve."""

    rank: int
    numa_node: int
    original: tuple[Chunk, ...]
    chunks: list[Chunk]

    @property
    def name(self) -> str:
        return chr(ord("A") + self.rank)

    @property
    def result(self) -> list[Real]:
        return [value for chunk in self.chunks for value in chunk.values]


@dataclass
class RingBuffer:
    """Bounded FIFO: monotonic head/tail; physical slot = counter % capacity.

    This serial model represents publish/consume ordering, not real GPU fences.
    A slot stores one simulated chunk, not a fixed number of hardware bytes.
    """

    capacity: int = 2
    head: int = field(default=0, init=False)
    tail: int = field(default=0, init=False)
    slots: list[Chunk | None] = field(init=False)

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("Buffer capacity must be a positive integer")
        self.slots = [None] * self.capacity

    def push(self, chunk: Chunk) -> int:
        if self.tail - self.head >= self.capacity:
            raise BufferError("Buffer full: wait for receiver head")
        slot = self.tail % self.capacity
        self.slots[slot] = chunk       # Write payload first.
        self.tail += 1                 # Then publish availability.
        return slot

    def pop(self) -> Chunk:
        if self.head == self.tail:
            raise BufferError("Buffer empty: wait for sender tail")
        slot = self.head % self.capacity
        chunk = self.slots[slot]
        assert chunk is not None
        self.slots[slot] = None
        self.head += 1                 # Reuse is allowed after consumption.
        return chunk


@dataclass
class PCIeLink:
    """Directed logical P2P connection, with a receiver-side FIFO."""

    source: int
    target: int
    route: str
    buffer: RingBuffer
    p2p_enabled: bool = True
    elements_sent: int = 0

    def send(self, payload: Chunk) -> int:
        if not self.p2p_enabled:
            raise RuntimeError("P2P unavailable: this simulator has no host fallback")
        slot = self.buffer.push(payload)
        self.elements_sent += len(payload.values)
        return slot


class GPUContainer:
    """Own four GPU objects, four links, the scheduler and round history.

    Default ring: A(0) -> D(3) -> C(2) -> B(1) -> A(0).
    The illustrative NUMA placement is A/B on node 0, C/D on node 1.
    All four edges are assumed to support P2P, including cross-NUMA edges.
    This is a selected teaching topology, not a probe of the current computer.
    """

    def __init__(self, inputs: Sequence[Sequence[Real]],
                 ring: Sequence[int] = (0, 3, 2, 1),
                 numa_nodes: Sequence[int] = (0, 0, 1, 1),
                 buffer_slots: int = 2):
        if len(inputs) != 4:
            raise ValueError("Exactly four GPU input vectors are required")
        if len(ring) != 4 or any(type(r) is not int for r in ring) or set(ring) != set(range(4)):
            raise ValueError("Ring must be a permutation of ranks 0, 1, 2, 3")
        if len(numa_nodes) != 4 or any(type(n) is not int or n < 0 for n in numa_nodes):
            raise ValueError("Four nonnegative NUMA node IDs are required")
        size = len(inputs[0])
        if size == 0 or any(len(row) != size for row in inputs):
            raise ValueError("Input vectors must have the same nonzero length")
        for row in inputs:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("Input values must be real numbers (not bool)")
                if not isinstance(value, int) and not math.isfinite(value):
                    raise ValueError("Input values must be finite")
        RingBuffer(buffer_slots)  # Validate even before constructing links.
        self.inputs = tuple(tuple(row) for row in inputs)
        self.ring = tuple(ring)
        self.numa_nodes = tuple(numa_nodes)
        self.buffer_slots = buffer_slots
        self.size = size
        # Uneven inputs are zero-padded, then trimmed from public results.
        self.chunk_size = (size + 3) // 4
        self.padded_size = 4 * self.chunk_size
        self.reset()

    def reset(self) -> None:
        self.gpus: dict[int, GPU] = {}
        for rank, row in enumerate(self.inputs):
            padded = row + (0,) * (self.padded_size - self.size)
            chunks = tuple(Chunk(k, padded[k*self.chunk_size:(k+1)*self.chunk_size],
                                 frozenset({rank})) for k in range(4))
            self.gpus[rank] = GPU(rank, self.numa_nodes[rank], chunks, list(chunks))
        self.links: dict[int, PCIeLink] = {}
        for pos, source in enumerate(self.ring):
            target = self.ring[(pos + 1) % 4]
            n0, n1 = self.numa_nodes[source], self.numa_nodes[target]
            route = (f"PCIe / NUMA {n0}" if n0 == n1 else
                     f"PCIe / NUMA {n0} -> CPU interconnect -> NUMA {n1} / PCIe")
            self.links[source] = PCIeLink(source, target, route, RingBuffer(self.buffer_slots))
        self.round = 0
        self.history = [self.snapshot("Initial", [])]

    @property
    def done(self) -> bool:
        return self.round == 6

    @property
    def expected(self) -> list[Real]:
        return [sum(row[i] for row in self.inputs) for i in range(self.size)]

    @property
    def results(self) -> dict[int, list[Real]]:
        if not self.done:
            raise RuntimeError("AllReduce is not complete")
        return {r: gpu.result[:self.size] for r, gpu in self.gpus.items()}

    def snapshot(self, phase: str, transfers: list[dict]) -> dict:
        return {"round": self.round, "phase": phase,
                "gpus": {str(r): {"name": g.name, "numa": g.numa_node,
                                   "chunks": [c.to_dict() for c in g.chunks]}
                         for r, g in self.gpus.items()},
                "transfers": transfers,
                "links": [{"source": link.source, "target": link.target,
                           "head": link.buffer.head, "tail": link.buffer.tail,
                           "capacity": link.buffer.capacity,
                           "elements_sent": link.elements_sent, "route": link.route}
                          for link in self.links.values()]}

    def step(self) -> dict:
        """Perform one concurrent logical round: all sends, then all receives."""
        if self.done:
            raise StopIteration("All six rounds have completed")
        # Fail before mutating the round when an edge cannot accept a packet.
        for link in self.links.values():
            if not link.p2p_enabled:
                raise RuntimeError("P2P unavailable; no host-memory fallback is modelled")
            if link.buffer.tail != link.buffer.head:
                raise RuntimeError("Round boundary requires empty link buffers")
        step = self.round + 1
        reducing = step <= 3
        transfers = []
        # At RS round 3, each rank owns the block with its own rank index.
        # Scheduling in ring-position space also supports other permutations.
        for pos, source in enumerate(self.ring):
            offset = step if reducing else step - 4
            chunk_index = self.ring[(pos - offset) % 4]
            link = self.links[source]
            payload = self.gpus[source].chunks[chunk_index]
            if not reducing and len(payload.contributors) != 4:
                raise AssertionError("AllGather must forward a fully reduced block")
            before = {"head": link.buffer.head, "tail": link.buffer.tail}
            slot = link.send(payload)
            transfers.append({"source": source, "target": link.target,
                              "chunk": chunk_index, "payload": payload.to_dict(),
                              "route": link.route, "slot": slot,
                              "operation": "recvReduce" if reducing else "recvCopy",
                              "before": before,
                              "published": {"head": link.buffer.head, "tail": link.buffer.tail}})
        # Crucial: no receive is applied until all send payloads are captured.
        for event in transfers:
            link = self.links[event["source"]]
            payload = link.buffer.pop()
            gpu = self.gpus[link.target]
            result = (payload.reduce_with(gpu.original[payload.index]) if reducing else payload)
            gpu.chunks[payload.index] = result
            event["result"] = result.to_dict()
            event["consumed"] = {"head": link.buffer.head, "tail": link.buffer.tail}
        self.round = step
        state = self.snapshot("ReduceScatter" if reducing else "AllGather", transfers)
        self.history.append(state)
        return state

    def run(self) -> dict[int, list[Real]]:
        while not self.done:
            self.step()
        return self.results

    def verify(self) -> bool:
        if not self.done:
            return False
        return all((a == b if isinstance(a, int) and isinstance(b, int)
                    else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))
                   for values in self.results.values()
                   for a, b in zip(values, self.expected))

    def trace(self) -> dict:
        return {"model": "CPU simulation; assumed P2P; no hardware performance model",
                "ring": list(self.ring), "input_length": self.size,
                "chunk_size": self.chunk_size, "padded_length": self.padded_size,
                "inputs": self.inputs, "expected": self.expected,
                "verified": self.verify(), "history": self.history}


def demo_inputs() -> list[list[int]]:
    return [[(rank + 1) * 10 + i for i in range(8)] for rank in range(4)]
