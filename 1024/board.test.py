import unittest
from Board import Board
import numpy as np
import random

board = Board()


def sample(board: Board, n: int = 10000, valid=True):
    round_counts: list[int] = []
    rewards: list[int] = []
    boards = []
    for epoch in range(n):
        board.reset()
        done = False
        count = 0
        while not done:
            action = random.choice([0, 1, 2, 3])
            if valid:
                while not board.squeezable(board.choice_to_direction(action)):
                    action = random.choice([0, 1, 2, 3])

            next_board, reward, done = board.step(action)
            rewards.append(reward)
            count += 1
            if done : boards.append(next_board)
        round_counts.append(count)
    return round_counts, rewards, boards


class Test(unittest.TestCase):

    def test_clear(self):
        board.clear()
        self.assertTrue(np.array_equal(
            board.board, np.zeros(board.board.shape)))

    def test_reset(self):
        board.reset()
        self.assertTrue(np.max(board.board) != 0)
        self.assertTrue(len(board.board.nonzero()) == 2)

    def test_is_done(self):
        done_case1 = np.array(
            [[2, 4, 16, 8], [4, 2, 8, 16], [2, 4, 2, 4], [4, 2, 8, 16]])
        done_case2 = np.array([[4, 8, 2, 4], [8, 256, 4, 64], [
                              2, 64, 32, 4], [8, 32, 8, 2]])
        self.assertTrue(board._is_done(done_case1))
        self.assertTrue(board._is_done(done_case2))

    def test_valid_move_until_end(self):
        round_counts, rewards, boards = sample(board, 100)
        for b in boards:
            self.assertTrue(board._is_done(b))



if __name__ == '__main__':
    unittest.main()
