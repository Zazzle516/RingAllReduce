import itertools
import json
import random
import unittest

from simulator import GPUContainer, SimpleBuffer, Slice, demo_inputs


class AlgorithmTests(unittest.TestCase):
    def test_default_two_channels_and_traffic(self):
        c = GPUContainer(demo_inputs(), record_history=False)
        result = c.run()
        self.assertTrue(c.verify())
        self.assertTrue(all(row == [100+4*i for i in range(32)] for row in result.values()))
        self.assertEqual([(ch.start, ch.stop) for ch in c.channels], [(0,16), (16,32)])
        self.assertEqual(c.counts['SEND'], 192)
        self.assertEqual(c.counts['RECV_REDUCE'], 96)
        self.assertEqual(c.counts['RECV_COPY'], 96)
        for kind in ['WAIT_FULL','WAIT_EMPTY','WAIT_DELAY','WAIT_DEPENDENCY']:
            self.assertGreater(c.counts[kind], 0)
        for ch in c.channels:
            for link in ch.links.values():
                self.assertTrue(link.buffer.empty)
                self.assertEqual((link.buffer.head,link.buffer.tail), (24,24))
                self.assertEqual(link.elements_sent, 6*ch.chunk_size)
            for rank in range(4):
                self.assertTrue(all(p.contributors == frozenset(range(4))
                                    for chunk in ch.data[rank] for p in chunk))

    def test_all_ring_orders_channels_padding_and_tail_slices(self):
        rng = random.Random(123)
        for size, channels in itertools.product([1,5,17], [1,2,3]):
            data = [[rng.randint(-100,100) for _ in range(size)] for _ in range(4)]
            expected = list(map(sum, zip(*data)))
            for ring in itertools.permutations(range(4)):
                with self.subTest(size=size, channels=channels, ring=ring):
                    c = GPUContainer(data, ring=ring, nchannels=channels,
                                     slice_elements=2, buffer_slots=1, receive_period=3,
                                     record_history=False)
                    self.assertTrue(all(row == expected for row in c.run().values()))
                    covered = [i for ch in c.channels for i in range(ch.start,ch.stop)]
                    self.assertEqual(covered, list(range(size)))

    def test_full_wait_does_not_overwrite_or_advance_sender(self):
        c = GPUContainer(demo_inputs(), record_history=False)
        full_waits = 0
        while not c.done:
            before = {(ch.id,r):(l.send_round,l.send_part,l.buffer.head,l.buffer.tail,
                                 tuple(l.buffer.slots)) for ch in c.channels for r,l in ch.links.items()}
            e = c.step_event()['event']
            if e['kind'] == 'WAIT_FULL':
                link = c.channels[e['channel']].links[e['source']]
                after = (link.send_round,link.send_part,link.buffer.head,link.buffer.tail,
                         tuple(link.buffer.slots))
                self.assertEqual(before[(e['channel'],e['source'])], after)
                full_waits += 1
            for ch in c.channels:
                for l in ch.links.values():
                    self.assertLessEqual(l.buffer.tail-l.buffer.head,l.buffer.capacity)
                    self.assertGreaterEqual(l.buffer.tail-l.buffer.head,0)
                    self.assertEqual(sum(p is not None for p in l.buffer.slots), l.buffer.tail-l.buffer.head)
        self.assertGreater(full_waits,0)

    def test_channel_progress_is_independent_and_slices_pipeline(self):
        c = GPUContainer(demo_inputs(), receive_period=1, receive_periods={(0,3):7}, record_history=False)
        diverged = pipeline = False
        while not c.done:
            e = c.step_event()['event']
            if c.channels[1].completed_round > c.channels[0].completed_round:
                diverged = True
            ch = c.channels[e['channel']]
            if e['kind'] == 'SEND' and e['round'] > ch.completed_round+1:
                pipeline = True
        self.assertTrue(diverged)
        self.assertTrue(pipeline)
        self.assertTrue(c.verify())

    def test_dependency_and_allgather_payloads(self):
        c = GPUContainer(demo_inputs(), nchannels=1, slice_elements=3,
                         receive_period=2, record_history=False)
        while not c.done:
            e = c.step_event()['event']
            if e['kind'] == 'SEND':
                self.assertEqual(len(e['payload']['contributors']), min(e['round'],4))
                self.assertLessEqual(len(e['payload']['values']),3)
            elif e['kind'] == 'RECV_COPY':
                self.assertEqual(e['payload']['values'],e['result']['values'])
                self.assertEqual(e['result']['contributors'],[0,1,2,3])
            elif e['kind'].startswith('WAIT_'):
                self.assertEqual(e['before'],e['after'])

    def test_history_is_immutable_json_and_round_milestones(self):
        c = GPUContainer([[r*10+i for i in range(8)] for r in range(4)], receive_period=2)
        first = c.step_event()
        saved = json.dumps(first,sort_keys=True)
        self.assertEqual(c.round,0)
        c.step_round()
        self.assertGreaterEqual(c.round,1)
        c.run()
        self.assertEqual(saved,json.dumps(first,sort_keys=True))
        trace = json.loads(json.dumps(c.trace()))
        self.assertTrue(trace['verified'])
        self.assertEqual(len(trace['history']),c.event_id+1)
        self.assertEqual(trace['schema_version'],2)
        with self.assertRaises(StopIteration):
            c.step()
        c.reset()
        self.assertEqual(c.event_id,0)
        self.assertEqual(len(c.history),1)
        self.assertEqual(c.counts,{})
        with self.assertRaises(RuntimeError):
            _ = c.results

    def test_original_input_and_float(self):
        values = [[r*0.2,-r*0.3,1.0] for r in range(4)]
        c = GPUContainer(values,record_history=False)
        values[0][0] = 999
        c.run()
        self.assertTrue(c.verify())
        self.assertAlmostEqual(c.results[0][0],1.2)

    def test_p2p_failure_does_not_mutate_scheduler(self):
        c = GPUContainer(demo_inputs(),record_history=False)
        c.channels[1].links[3].p2p_enabled = False
        with self.assertRaises(RuntimeError):
            c.step_event()
        self.assertEqual((c.event_id,c.cursor,c.sweep),(0,0,0))

    def test_invalid_inputs_and_configuration(self):
        bad = [[], [[1]]*3, [[1],[1],[1],[1,2]], [[]]*4, [[True]]*4,
               [['x']]*4, [[float('inf')]]*4, [[float('nan')]]*4]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                GPUContainer(data)
        for kwargs in [{'nchannels':0},{'nchannels':True},{'buffer_slots':0},
                       {'slice_elements':-1},{'receive_period':0},{'slow_rank':4},
                       {'ring':(0,1,2,2)},{'numa_nodes':(0,1)},
                       {'receive_periods':{(2,3):2}},{'receive_periods':{(0,3):0}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GPUContainer(demo_inputs(),**kwargs)


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
