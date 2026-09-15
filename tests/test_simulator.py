import itertools
import random
import unittest

from simulator import Chunk, GPUContainer, RingBuffer, demo_inputs


class AlgorithmTests(unittest.TestCase):
    def test_demo_and_matching_diagram_schedule(self):
        sim = GPUContainer(demo_inputs())
        first = sim.step()
        self.assertEqual([t['chunk'] for t in first['transfers']], [1, 0, 3, 2])
        self.assertEqual([len(t['result']['contributors']) for t in first['transfers']], [2]*4)
        sim.step()
        state = sim.step()
        for rank in range(4):
            chunk = sim.gpus[rank].chunks[rank]
            self.assertEqual(chunk.contributors, frozenset(range(4)))
            self.assertEqual(list(chunk.values), sim.expected[rank*2:rank*2+2])
        original_history = repr(state)
        sim.run()
        self.assertEqual(repr(state), original_history)
        self.assertTrue(sim.verify())
        self.assertEqual(sim.results[0], [100, 104, 108, 112, 116, 120, 124, 128])
        for link in sim.links.values():
            self.assertEqual((link.buffer.head, link.buffer.tail), (6, 6))
            self.assertEqual(link.elements_sent, 12)
        self.assertEqual(len(sim.history), 7)
        with self.assertRaises(StopIteration):
            sim.step()

    def test_all_ring_permutations_and_uneven_vectors(self):
        rng = random.Random(17)
        for length in [1, 2, 3, 4, 5, 8, 17, 100]:
            inputs = [[rng.randint(-1000, 1000) for _ in range(length)] for _ in range(4)]
            expected = [sum(row[i] for row in inputs) for i in range(length)]
            for ring in itertools.permutations(range(4)):
                with self.subTest(length=length, ring=ring):
                    sim = GPUContainer(inputs, ring=ring, buffer_slots=1)
                    sim.run()
                    self.assertTrue(sim.verify())
                    self.assertTrue(all(row == expected for row in sim.results.values()))
                    for state in sim.history[4:]:
                        self.assertTrue(all(len(t['payload']['contributors']) == 4 for t in state['transfers']))

    def test_float_and_input_copy(self):
        inputs = [[0.1 * (r+1), -0.3 * r, 0.0] for r in range(4)]
        sim = GPUContainer(inputs)
        inputs[0][0] = 999
        sim.run()
        self.assertTrue(sim.verify())
        self.assertAlmostEqual(sim.results[0][0], 1.0)

    def test_reset(self):
        sim = GPUContainer(demo_inputs())
        sim.run()
        sim.reset()
        self.assertEqual(sim.round, 0)
        self.assertEqual(len(sim.history), 1)
        self.assertFalse(sim.verify())
        with self.assertRaises(RuntimeError):
            _ = sim.results
        sim.run()
        self.assertTrue(sim.verify())

    def test_disabled_p2p_is_atomic(self):
        sim = GPUContainer(demo_inputs())
        sim.links[3].p2p_enabled = False
        with self.assertRaises(RuntimeError):
            sim.step()
        self.assertEqual(sim.round, 0)
        self.assertTrue(all(link.elements_sent == 0 for link in sim.links.values()))

    def test_invalid_inputs(self):
        bad = [[], [[1]]*3, [[1], [1], [1], [1, 2]], [[], [], [], []],
               [[True]]*4, [["1"]]*4, [[float('nan')]]*4, [[float('inf')]]*4]
        for inputs in bad:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                GPUContainer(inputs)
        for kwargs in [{'ring': (0, 1, 2, 2)}, {'numa_nodes': (0, 1)}, {'buffer_slots': 0}]:
            with self.assertRaises(ValueError):
                GPUContainer(demo_inputs(), **kwargs)


class BufferTests(unittest.TestCase):
    def test_full_empty_fifo_and_wraparound(self):
        buffer = RingBuffer(2)
        first = Chunk(0, (10,), frozenset({0}))
        second = Chunk(1, (20,), frozenset({1}))
        with self.assertRaises(BufferError):
            buffer.pop()
        self.assertEqual(buffer.push(first), 0)
        self.assertEqual(buffer.push(second), 1)
        with self.assertRaises(BufferError):
            buffer.push(first)
        self.assertEqual(buffer.pop(), first)
        self.assertEqual(buffer.push(first), 0)
        self.assertEqual(buffer.pop(), second)
        self.assertEqual(buffer.pop(), first)
        self.assertEqual((buffer.head, buffer.tail), (3, 3))

    def test_duplicate_contribution_rejected(self):
        chunk = Chunk(0, (1,), frozenset({0}))
        with self.assertRaises(ValueError):
            chunk.reduce_with(chunk)


if __name__ == '__main__':
    unittest.main()
