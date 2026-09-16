"""Deterministic event simulation, not real PCIe, DMA or GPU memory ordering."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping, Sequence


def positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Slice:
    """Transport payload: one piece of a channel's algorithm chunk."""
    channel: int
    chunk: int
    part: int
    offset: int
    values: tuple[Real, ...]
    contributors: frozenset[int]
    round: int = 0

    def to_dict(self) -> dict:
        return {"channel": self.channel, "chunk": self.chunk, "part": self.part,
                "offset": self.offset, "values": list(self.values),
                "contributors": sorted(self.contributors), "round": self.round}


@dataclass
class SimpleBuffer:
    """Receiver-owned FIFO. One slot holds at most slot_elements values.

    Counters are monotonic: slots wrap, head/tail never wrap. push publishes
    tail after writing; pop releases head after reading. This serial ordering
    illustrates the protocol but does not implement hardware memory fences.
    """
    capacity: int = 2
    slot_elements: int = 1
    head: int = field(default=0, init=False)
    tail: int = field(default=0, init=False)
    slots: list[Slice | None] = field(init=False)

    def __post_init__(self):
        positive(self.capacity, "buffer_slots")
        positive(self.slot_elements, "slice_elements")
        self.slots = [None] * self.capacity

    @property
    def full(self) -> bool:
        return self.tail - self.head == self.capacity

    @property
    def empty(self) -> bool:
        return self.tail == self.head

    def push(self, packet: Slice) -> int:
        if not 0 < len(packet.values) <= self.slot_elements:
            raise ValueError("Slice exceeds slot capacity or is empty")
        if self.full:
            raise BufferError("WAIT_FULL: receiver has not released head")
        slot = self.tail % self.capacity
        assert self.slots[slot] is None
        self.slots[slot] = packet
        self.tail += 1
        return slot

    def peek(self) -> Slice:
        if self.empty:
            raise BufferError("WAIT_EMPTY: sender has not published tail")
        packet = self.slots[self.head % self.capacity]
        assert packet is not None
        return packet

    def pop(self) -> Slice:
        packet = self.peek()
        self.slots[self.head % self.capacity] = None
        self.head += 1
        return packet

    def to_dict(self) -> dict:
        return {"head": self.head, "tail": self.tail, "capacity": self.capacity,
                "slot_elements": self.slot_elements, "occupied": self.tail-self.head,
                "slots": [p.to_dict() if p else None for p in self.slots]}


@dataclass
class GPU:
    rank: int
    numa_node: int
    input_values: tuple[Real, ...]
    channels: dict[int, GPUChannelState] = field(default_factory=dict, init=False)

    @property
    def name(self):
        return chr(65 + self.rank)


@dataclass
class PCIeLink:
    """One channel's directed logical connection, not an exclusive PCIe lane."""
    source: int
    target: int
    route: str
    buffer: SimpleBuffer
    receive_period: int = 1
    p2p_enabled: bool = True
    send_round: int = 1
    send_part: int = 0
    recv_round: int = 1
    recv_part: int = 0
    elements_sent: int = 0
    send_status: str = "READY"
    recv_status: str = "READY"


@dataclass(frozen=True)
class Channel:
    """Immutable partition and ring plan; device data lives on each GPU."""
    id: int
    start: int
    stop: int
    ring: tuple[int, ...]
    slice_elements: int

    @property
    def chunk_size(self):
        return (self.stop - self.start + 3) // 4

    @property
    def padded_size(self):
        return 4 * self.chunk_size

    @property
    def parts(self):
        return (self.chunk_size + self.slice_elements - 1) // self.slice_elements

    def chunk_index(self, source, round_id):
        offset = round_id if round_id <= 3 else round_id - 4
        return self.ring[(self.ring.index(source) - offset) % 4]


@dataclass
class GPUChannelState:
    """One GPU's local memory and connections for one channel.

    recv_buffer belongs here. The upstream send_link and this recv_link
    reference the SAME connection, whose buffer references this recv_buffer.
    """
    original: tuple[Real, ...]
    data: list[list[Slice]]
    recv_buffer: SimpleBuffer
    send_link: PCIeLink | None = None
    recv_link: PCIeLink | None = None

    @classmethod
    def create(cls, gpu, channel, buffer_slots):
        values = gpu.input_values[channel.start:channel.stop]
        values += (0,) * (channel.padded_size - len(values))
        chunks = []
        for chunk in range(4):
            slices = []
            for part, offset in enumerate(range(0, channel.chunk_size, channel.slice_elements)):
                first = chunk * channel.chunk_size + offset
                end = min(first + channel.slice_elements, (chunk + 1) * channel.chunk_size)
                slices.append(Slice(channel.id, chunk, part, offset,
                                    values[first:end], frozenset({gpu.rank})))
            chunks.append(slices)
        return cls(values, chunks, SimpleBuffer(buffer_slots, channel.slice_elements))

    def attempt(self, channel, action, tick):
        """Execute one local send or receive attempt; return an event."""
        link = self.send_link if action == "send" else self.recv_link
        assert link is not None
        source = link.source
        b = link.buffer
        rnd = link.send_round if action == "send" else link.recv_round
        if rnd > 6:
            return None
        event = {"channel": channel.id, "source": source, "target": link.target,
                 "actor": action, "round": rnd, "tick": tick,
                 "phase": "ReduceScatter" if rnd <= 3 else "AllGather",
                 "before": {"head": b.head, "tail": b.tail}}
        if action == "send":
            k, part = channel.chunk_index(source, rnd), link.send_part
            event.update(chunk=k, part=part)
            packet = self.data[k][part]
            if b.full:
                event["kind"] = "WAIT_FULL"
            elif rnd > 1 and packet.round != rnd-1:
                event["kind"] = "WAIT_DEPENDENCY"
            else:
                assert len(packet.contributors) == min(rnd, 4)
                outgoing = Slice(channel.id, k, part, packet.offset, packet.values, packet.contributors, rnd)
                event.update(kind="SEND", slot=b.push(outgoing), payload=outgoing.to_dict())
                link.elements_sent += len(outgoing.values)
                link.send_part += 1
                if link.send_part == channel.parts:
                    link.send_part = 0
                    link.send_round += 1
            link.send_status = event["kind"]
        else:
            event.update(chunk=channel.chunk_index(source, rnd), part=link.recv_part)
            if b.empty:
                event["kind"] = "WAIT_EMPTY"
            elif tick % link.receive_period:
                event["kind"] = "WAIT_DELAY"
            else:
                packet = b.peek()
                assert (packet.round, packet.part, packet.chunk) == (rnd, link.recv_part, event["chunk"])
                values, contributors = packet.values, packet.contributors
                if rnd <= 3:
                    assert link.target not in contributors
                    begin = packet.chunk*channel.chunk_size + packet.offset
                    local = self.original[begin:begin+len(values)]
                    values = tuple(a+v for a, v in zip(values, local))
                    contributors = contributors | {link.target}
                else:
                    assert len(contributors) == 4
                result = Slice(channel.id, packet.chunk, packet.part, packet.offset,
                               values, frozenset(contributors), rnd)
                self.data[packet.chunk][packet.part] = result
                slot = b.head % b.capacity
                b.pop()  # Release only after consuming and storing the result.
                event.update(kind="RECV_REDUCE" if rnd <= 3 else "RECV_COPY",
                             slot=slot, payload=packet.to_dict(), result=result.to_dict())
                link.recv_part += 1
                if link.recv_part == channel.parts:
                    link.recv_part = 0
                    link.recv_round += 1
            link.recv_status = event["kind"]
        event["after"] = {"head": b.head, "tail": b.tail}
        return event


class Scheduler:
    """Fair single-thread event scheduler; has no reference to RingAllReduceSimulator.

    It borrows GPUs and channel plans, advancing their local state one attempt
    at a time. A sweep is a scheduling unit, not hardware time or a barrier.
    """
    def __init__(self, gpus, channels):
        self.gpus, self.channels = gpus, channels
        self.actors = [(c.id, r, action) for action in ("send", "recv")
                       for r in channels[0].ring for c in channels]
        self.cursor = self.sweep = self.event_id = 0
        self.last_event = None
        self.counts = {}

    def links(self, channel_id):
        for source in self.channels[channel_id].ring:
            yield self.gpus[source].channels[channel_id].send_link

    def completed_round(self, channel_id):
        return min(link.recv_round - 1 for link in self.links(channel_id))

    @property
    def done(self):
        return all(self.completed_round(c.id) == 6 for c in self.channels)

    @property
    def round(self):
        return min(self.completed_round(c.id) for c in self.channels)

    def step_event(self):
        if self.done:
            raise StopIteration("All channels have completed AllReduce")
        if any(not link.p2p_enabled for c in self.channels for link in self.links(c.id)):
            raise RuntimeError("P2P unavailable; no host fallback is modelled")
        while True:
            c, source, action = self.actors[self.cursor]
            tick = self.sweep
            self.cursor = (self.cursor + 1) % len(self.actors)
            if self.cursor == 0:
                self.sweep += 1
            sender = self.gpus[source].channels[c]
            rank = source if action == "send" else sender.send_link.target
            event = self.gpus[rank].channels[c].attempt(self.channels[c], action, tick)
            if event is not None:
                break
        self.event_id += 1
        event["id"] = self.event_id
        self.last_event = event
        self.counts[event["kind"]] = self.counts.get(event["kind"], 0) + 1
        return event


class RingAllReduceSimulator:
    """Assemble the simulation, manage its lifecycle, and expose GUI/CLI APIs.

    GPUs own local channel state. Scheduler advances communication events.
    This facade records snapshots, assembles results and verifies them.

    Each scheduler sweep offers every active sender/receiver one turn. A slow
    receiver consumes only on sweeps divisible by its receive_period. One
    call to step_event emits one send, receive or wait. No global round barrier
    exists. step_round is only a convenient UI fast-forward over these events.
    """
    def __init__(self, inputs: Sequence[Sequence[Real]], ring: Sequence[int] = (0, 3, 2, 1),
                 numa_nodes: Sequence[int] = (0, 0, 1, 1), buffer_slots: int = 2,
                 nchannels: int = 2, slice_elements: int = 1,
                 slow_rank: int | None = 3, receive_period: int = 4,
                 receive_periods: Mapping[tuple[int, int], int] | None = None,
                 record_history: bool = True):
        if len(inputs) != 4:
            raise ValueError("Exactly four input vectors are required")
        if len(ring) != 4 or any(type(r) is not int for r in ring) or set(ring) != set(range(4)):
            raise ValueError("ring must be a permutation of 0, 1, 2, 3")
        if len(numa_nodes) != 4 or any(type(n) is not int or n < 0 for n in numa_nodes):
            raise ValueError("Four nonnegative NUMA IDs are required")
        size = len(inputs[0])
        if not size or any(len(row) != size for row in inputs):
            raise ValueError("Input vectors must have the same nonzero length")
        for row in inputs:
            for v in row:
                if isinstance(v, bool) or not isinstance(v, Real):
                    raise ValueError("Input values must be real numbers, not bool")
                if not isinstance(v, int) and not math.isfinite(v):
                    raise ValueError("Input values must be finite")
        for name, value in [("nchannels", nchannels), ("buffer_slots", buffer_slots),
                            ("slice_elements", slice_elements), ("receive_period", receive_period)]:
            positive(value, name)
        if slow_rank is not None and (type(slow_rank) is not int or slow_rank not in range(4)):
            raise ValueError("slow_rank must be None or a rank from 0 to 3")
        self.receive_periods = dict(receive_periods or {})
        for key, value in self.receive_periods.items():
            if (not isinstance(key, tuple) or len(key) != 2 or
                    any(type(k) is not int for k in key) or key[0] not in range(nchannels) or key[1] not in range(4)):
                raise ValueError("receive_periods keys must be (channel_id, receiver_rank)")
            positive(value, "receive_periods value")
        self.inputs = tuple(tuple(row) for row in inputs)
        self.ring, self.numa_nodes = tuple(ring), tuple(numa_nodes)
        self.size, self.nchannels = size, nchannels
        self.buffer_slots, self.slice_elements = buffer_slots, slice_elements
        self.slow_rank, self.receive_period = slow_rank, receive_period
        self.record_history = record_history
        self.reset()

    def reset(self):
        self.gpus = {r: GPU(r, self.numa_nodes[r], row) for r, row in enumerate(self.inputs)}
        width, extra = divmod(self.size, self.nchannels)
        self.channels = []
        start = 0
        for c in range(self.nchannels):
            stop = start + width + (c < extra)
            self.channels.append(Channel(c, start, stop, self.ring, self.slice_elements))
            start = stop
        self.padded_size = sum(c.padded_size for c in self.channels)
        for gpu in self.gpus.values():
            gpu.channels = {c.id: GPUChannelState.create(gpu, c, self.buffer_slots)
                            for c in self.channels}
        for c in self.channels:
            for pos, source in enumerate(c.ring):
                target = c.ring[(pos + 1) % 4]
                sender = self.gpus[source].channels[c.id]
                receiver = self.gpus[target].channels[c.id]
                n0, n1 = self.gpus[source].numa_node, self.gpus[target].numa_node
                route = (f"NUMA {n0} / PCIe" if n0 == n1 else
                         f"NUMA {n0} -> CPU interconnect -> NUMA {n1}")
                period = self.receive_periods.get((c.id, target),
                             self.receive_period if target == self.slow_rank else 1)
                link = PCIeLink(source, target, route, receiver.recv_buffer, period)
                if not c.parts:
                    link.send_round = link.recv_round = 7
                    link.send_status = link.recv_status = "DONE"
                sender.send_link = receiver.recv_link = link
        self.scheduler = Scheduler(self.gpus, self.channels)
        self.history = [self.snapshot()]

    @property
    def done(self):
        return self.scheduler.done

    @property
    def round(self):
        """Minimum fully received round across active channels; for display only."""
        return self.scheduler.round

    @property
    def event_id(self):
        return self.scheduler.event_id

    @property
    def last_event(self):
        return self.scheduler.last_event

    @property
    def counts(self):
        return self.scheduler.counts

    @property
    def cursor(self):
        return self.scheduler.cursor

    @property
    def sweep(self):
        return self.scheduler.sweep

    @property
    def expected(self):
        return [sum(row[i] for row in self.inputs) for i in range(self.size)]

    @property
    def results(self):
        if not self.done:
            raise RuntimeError("AllReduce is not complete")
        result = {}
        for rank in range(4):
            result[rank] = []
            for c in self.channels:
                values = [v for chunk in self.gpus[rank].channels[c.id].data for packet in chunk for v in packet.values]
                result[rank].extend(values[:c.stop-c.start])
        return result

    def step_event(self):
        """Advance the scheduler, then record a GUI/JSON snapshot."""
        event = self.scheduler.step_event()
        if self.record_history:
            state = self.snapshot()
            self.history.append(state)
            return state
        return {"event_id": self.event_id, "event": event, "round": self.round, "done": self.done}

    def step(self):
        """Version 2: one event. Use step_round() for coarse algorithm progress."""
        return self.step_event()

    def step_round(self):
        if self.done:
            raise StopIteration("All channels have completed AllReduce")
        target = self.round+1
        state = None
        while self.round < target:
            state = self.step_event()
        return state

    def run(self):
        while not self.done:
            self.step_event()
        return self.results

    def verify(self):
        if not self.done:
            return False
        return all((a == b if isinstance(a, int) and isinstance(b, int)
                    else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))
                   for row in self.results.values() for a, b in zip(row, self.expected))

    def snapshot(self):
        channels = []
        for c in self.channels:
            devices = {}
            for rank, gpu in self.gpus.items():
                chunks = gpu.channels[c.id].data
                devices[str(rank)] = {"name": self.gpus[rank].name, "numa": self.numa_nodes[rank],
                    "chunks": [{"index": k, "values": [v for p in parts for v in p.values],
                                "slices": [p.to_dict() for p in parts],
                                "completed_slices": sum(len(p.contributors) == 4 for p in parts),
                                "complete": bool(parts) and all(len(p.contributors) == 4 for p in parts)}
                               for k, parts in enumerate(chunks)]}
            links = []
            for link in self.scheduler.links(c.id):
                links.append({"source": link.source, "target": link.target, "route": link.route,
                              "buffer": link.buffer.to_dict(), "elements_sent": link.elements_sent,
                              "send_round": link.send_round, "send_part": link.send_part,
                              "recv_round": link.recv_round, "recv_part": link.recv_part,
                              "receive_period": link.receive_period,
                              "send_status": link.send_status, "recv_status": link.recv_status})
            channels.append({"id": c.id, "start": c.start, "stop": c.stop,
                             "chunk_size": c.chunk_size, "parts": c.parts,
                             "completed_round": self.scheduler.completed_round(c.id), "gpus": devices, "links": links})
        # All nested mutable collections are new; history never shares live FIFO storage.
        import copy
        return {"event_id": self.event_id, "tick": self.last_event['tick'] if self.last_event else 0,
                "round": self.round, "done": self.done, "counts": dict(self.counts),
                "event": copy.deepcopy(self.last_event), "channels": channels}

    def trace(self):
        return {"schema_version": 2, "model": "CPU event simulation; no hardware performance estimate",
                "ring": list(self.ring), "inputs": self.inputs, "input_length": self.size,
                "padded_length": self.padded_size, "nchannels": self.nchannels,
                "buffer_slots": self.buffer_slots, "slice_elements": self.slice_elements,
                "slow_rank": self.slow_rank, "receive_period": self.receive_period,
                "receive_periods": [{"channel": c, "rank": r, "period": p}
                                    for (c, r), p in self.receive_periods.items()],
                "expected": self.expected, "verified": self.verify(),
                "history_recorded": self.record_history, "counts": dict(self.counts),
                "history": self.history, "final_state": self.snapshot()}


def demo_inputs():
    """32 elements -> two 16-element channels -> four 4-slice chunks each."""
    return [[(rank+1)*10+i for i in range(32)] for rank in range(4)]
