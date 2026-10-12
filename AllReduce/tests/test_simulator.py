import itertools
import json
import random
import unittest

from AllReduce.simulator import RingAllReduceSimulator, SimpleBuffer, Slice, demo_inputs


class AlgorithmTests(unittest.TestCase):
    def test_gpu_owned_buffers_and_reset_reconnects(self):
        simulator = RingAllReduceSimulator(demo_inputs())
        buffers = []
        for gpu in simulator.gpus.values():
            for channel_id, local in gpu.channels.items():
                upstream = simulator.gpus[local.recv_link.source].channels[channel_id]
                downstream = simulator.gpus[local.send_link.target].channels[channel_id]
                self.assertIs(upstream.send_link, local.recv_link)
                self.assertIs(local.recv_buffer, local.recv_link.buffer)
                self.assertIs(local.send_link.buffer, downstream.recv_buffer)
                buffers.append(local.recv_buffer)
        self.assertEqual(len({id(b) for b in buffers}), 8)
        simulator.step_event()
        self.assertEqual(simulator.gpus[3].channels[0].recv_buffer.tail, 1)
        old_scheduler = simulator.scheduler
        old_receiver = simulator.gpus[3].channels[0]
        simulator.reset()
        self.assertIsNot(simulator.scheduler, old_scheduler)
        self.assertEqual(old_receiver.recv_buffer.tail, 1)
        self.assertEqual(simulator.gpus[3].channels[0].recv_buffer.tail, 0)
        simulator.step_event()
        self.assertEqual(old_receiver.recv_buffer.tail, 1)
        self.assertEqual(simulator.gpus[3].channels[0].recv_buffer.tail, 1)
        simulator.run()
        self.assertTrue(simulator.verify())

    def test_default_two_channels_and_traffic(self):
        simulator = RingAllReduceSimulator(demo_inputs(), record_history=False)
        result = simulator.run()
        self.assertTrue(simulator.verify())
        self.assertTrue(all(row == [100+4*i for i in range(32)] for row in result.values()))
        self.assertEqual([(ch.start, ch.stop) for ch in simulator.channels], [(0,16), (16,32)])
        self.assertEqual(simulator.counts['SEND'], 192)
        self.assertEqual(simulator.counts['RECV_REDUCE'], 96)
        self.assertEqual(simulator.counts['RECV_COPY'], 96)
        for kind in ['WAIT_FULL','WAIT_EMPTY','WAIT_DELAY','WAIT_DEPENDENCY']:
            self.assertGreater(simulator.counts[kind], 0)
        for ch in simulator.channels:
            for link in simulator.scheduler.links(ch.id):
                self.assertTrue(link.buffer.empty)
                self.assertEqual((link.buffer.head,link.buffer.tail), (24,24))
                self.assertEqual(link.elements_sent, 6*ch.chunk_size)
            for rank in range(4):
                self.assertTrue(all(p.contributors == frozenset(range(4))
                                    for chunk in simulator.gpus[rank].channels[ch.id].data for p in chunk))

    def test_all_ring_orders_channels_padding_and_tail_slices(self):
        rng = random.Random(123)
        for size, channels in itertools.product([1,5,17], [1,2,3]):
            data = [[rng.randint(-100,100) for _ in range(size)] for _ in range(4)]
            expected = list(map(sum, zip(*data)))
            for ring in itertools.permutations(range(4)):
                with self.subTest(size=size, channels=channels, ring=ring):
                    simulator = RingAllReduceSimulator(data, ring=ring, nchannels=channels,
                                     slice_elements=2, buffer_slots=1, receive_period=3,
                                     record_history=False)
                    self.assertTrue(all(row == expected for row in simulator.run().values()))
                    covered = [i for ch in simulator.channels for i in range(ch.start,ch.stop)]
                    self.assertEqual(covered, list(range(size)))

    def test_full_wait_does_not_overwrite_or_advance_sender(self):
        simulator = RingAllReduceSimulator(demo_inputs(), record_history=False)
        full_waits = 0
        while not simulator.done:
            before = {(ch.id,r):(l.send_round,l.send_part,l.buffer.head,l.buffer.tail,
                                 tuple(l.buffer.slots)) for ch in simulator.channels for r in simulator.ring for l in [simulator.gpus[r].channels[ch.id].send_link]}
            e = simulator.step_event()['event']
            if e['kind'] == 'WAIT_FULL':
                link = simulator.gpus[e['source']].channels[e['channel']].send_link
                after = (link.send_round,link.send_part,link.buffer.head,link.buffer.tail,
                         tuple(link.buffer.slots))
                self.assertEqual(before[(e['channel'],e['source'])], after)
                full_waits += 1
            for ch in simulator.channels:
                for l in simulator.scheduler.links(ch.id):
                    self.assertLessEqual(l.buffer.tail-l.buffer.head,l.buffer.capacity)
                    self.assertGreaterEqual(l.buffer.tail-l.buffer.head,0)
                    self.assertEqual(sum(p is not None for p in l.buffer.slots), l.buffer.tail-l.buffer.head)
        self.assertGreater(full_waits,0)

    def test_channel_progress_is_independent_and_slices_pipeline(self):
        simulator = RingAllReduceSimulator(demo_inputs(), receive_period=1, receive_periods={(0,3):7}, record_history=False)
        diverged = pipeline = False
        while not simulator.done:
            e = simulator.step_event()['event']
            if simulator.scheduler.completed_round(1) > simulator.scheduler.completed_round(0):
                diverged = True
            ch = simulator.channels[e['channel']]
            if e['kind'] == 'SEND' and e['round'] > simulator.scheduler.completed_round(ch.id)+1:
                pipeline = True
        self.assertTrue(diverged)
        self.assertTrue(pipeline)
        self.assertTrue(simulator.verify())

    def test_dependency_and_allgather_payloads(self):
        simulator = RingAllReduceSimulator(demo_inputs(), nchannels=1, slice_elements=3,
                         receive_period=2, record_history=False)
        while not simulator.done:
            e = simulator.step_event()['event']
            if e['kind'] == 'SEND':
                self.assertEqual(len(e['payload']['contributors']), min(e['round'],4))
                self.assertLessEqual(len(e['payload']['values']),3)
            elif e['kind'] == 'RECV_COPY':
                self.assertEqual(e['payload']['values'],e['result']['values'])
                self.assertEqual(e['result']['contributors'],[0,1,2,3])
            elif e['kind'].startswith('WAIT_'):
                self.assertEqual(e['before'],e['after'])

    def test_history_is_immutable_json_and_round_milestones(self):
        simulator = RingAllReduceSimulator([[r*10+i for i in range(8)] for r in range(4)], receive_period=2)
        first = simulator.step_event()
        saved = json.dumps(first,sort_keys=True)
        self.assertEqual(simulator.round,0)
        simulator.step_round()
        self.assertGreaterEqual(simulator.round,1)
        simulator.run()
        self.assertEqual(saved,json.dumps(first,sort_keys=True))
        trace = json.loads(json.dumps(simulator.trace()))
        self.assertTrue(trace['verified'])
        self.assertEqual(len(trace['history']),simulator.event_id+1)
        self.assertEqual(trace['schema_version'],2)
        with self.assertRaises(StopIteration):
            simulator.step()
        simulator.reset()
        self.assertEqual(simulator.event_id,0)
        self.assertEqual(len(simulator.history),1)
        self.assertEqual(simulator.counts,{})
        with self.assertRaises(RuntimeError):
            _ = simulator.results

    def test_original_input_and_float(self):
        values = [[r*0.2,-r*0.3,1.0] for r in range(4)]
        simulator = RingAllReduceSimulator(values,record_history=False)
        values[0][0] = 999
        simulator.run()
        self.assertTrue(simulator.verify())
        self.assertAlmostEqual(simulator.results[0][0],1.2)

    def test_p2p_failure_does_not_mutate_scheduler(self):
        simulator = RingAllReduceSimulator(demo_inputs(),record_history=False)
        simulator.gpus[3].channels[1].send_link.p2p_enabled = False
        with self.assertRaises(RuntimeError):
            simulator.step_event()
        self.assertEqual((simulator.event_id,simulator.cursor,simulator.sweep),(0,0,0))

    def test_invalid_inputs_and_configuration(self):
        bad = [[], [[1]]*3, [[1],[1],[1],[1,2]], [[]]*4, [[True]]*4,
               [['x']]*4, [[float('inf')]]*4, [[float('nan')]]*4]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                RingAllReduceSimulator(data)
        for kwargs in [{'nchannels':0},{'nchannels':True},{'buffer_slots':0},
                       {'slice_elements':-1},{'receive_period':0},{'slow_rank':4},
                       {'ring':(0,1,2,2)},{'numa_nodes':(0,1)},
                       {'receive_periods':{(2,3):2}},{'receive_periods':{(0,3):0}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RingAllReduceSimulator(demo_inputs(),**kwargs)


class BufferTests(unittest.TestCase):
    def test_bounded_fifo_wraparound_and_capacity(self):
        b = SimpleBuffer(2,2)
        p0 = Slice(0,0,0,0,(1,2),frozenset({0}),1)
        p1 = Slice(0,0,1,2,(3,),frozenset({0}),1)
        with self.assertRaises(BufferError):
            b.pop()
        self.assertEqual(b.push(p0),0)
        self.assertEqual(b.push(p1),1)
        with self.assertRaises(BufferError):
            b.push(p0)
        self.assertEqual(b.pop(),p0)
        self.assertEqual(b.push(p0),0)
        self.assertEqual(b.pop(),p1)
        self.assertEqual(b.pop(),p0)
        self.assertEqual((b.head,b.tail),(3,3))
        with self.assertRaises(ValueError):
            b.push(Slice(0,0,0,0,(1,2,3),frozenset({0})))
        self.assertEqual((b.head,b.tail),(3,3))


if __name__ == '__main__':
    unittest.main()
