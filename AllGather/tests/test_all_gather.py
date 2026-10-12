"""独立 AllGather 的逐轮语义和兼容 API。"""
import itertools
import unittest


from AllGather.simulator import all_gather, gather_rounds


class AllGatherTests(unittest.TestCase):
    def test_rounds_move_each_block_one_hop_for_every_ring(self):
        inputs = ((10, 11), (20, 21), (30, 31), (40, 41))
        for ring in itertools.permutations(range(4)):
            with self.subTest(ring=ring):
                states = list(gather_rounds(inputs, ring))
                self.assertEqual(len(states), 4)
                for round_id, state in enumerate(states):
                    for rank, blocks in enumerate(state["blocks"]):
                        self.assertEqual(sum(b is not None for b in blocks), round_id + 1)
                        self.assertEqual(blocks[rank], inputs[rank])
                    for source, target, owner, block in state["transfers"]:
                        self.assertEqual(target, ring[(ring.index(source) + 1) % 4])
                        self.assertEqual(block, inputs[owner])
                        self.assertEqual(states[round_id - 1]["blocks"][source][owner], block)
                        self.assertIsNone(states[round_id - 1]["blocks"][target][owner])
                        self.assertEqual(state["blocks"][target][owner], block)
                self.assertTrue(all(blocks == inputs for blocks in states[-1]["blocks"]))
                self.assertEqual(all_gather(inputs, ring, verbose=False),
                                 [[10, 11, 20, 21, 30, 31, 40, 41]] * 4)


if __name__ == "__main__":
    unittest.main()
