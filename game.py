class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Start a new game with an empty board.
        """
        self.board = [' ' for _ in range(9)]
        return self.get_state()

    def get_state(self):
        """
        Returns the current board as a string for easy hashing.
        E.g. 'XO X O   '
        """
        return ''.join(self.board)

    def available_actions(self):
        """
        List of indices where the board is empty.
        """
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def step(self, action, player):
        """
        Executes a move by the given player.
        Returns (next_state, reward, done, winner)
        Reward from the perspective of the player making the move:
            +1 = player wins
            -1 = player loses (opponent has won)
             0 = draw or ongoing
        """
        if self.board[action] != ' ':
            raise ValueError(f"Invalid action: Cell {action} is already filled.")

        self.board[action] = player
        winner = self.check_winner()

        if winner == player:
            return self.get_state(), 1, True, winner
        elif winner is not None:
            return self.get_state(), -1, True, winner
        elif ' ' not in self.board:
            return self.get_state(), 0, True, None
        else:
            return self.get_state(), 0, False, None

    def check_winner(self):
        """
        Checks for a winning condition.
        Returns 'X', 'O', or None.
        """
        b = self.board
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for i, j, k in lines:
            if b[i] == b[j] == b[k] and b[i] != ' ':
                return b[i]
        return None

    def render(self):
        """
        Nicely prints the current board.
        """
        b = self.board
        print(f" {b[0]} | {b[1]} | {b[2]}")
        print("---+---+---")
        print(f" {b[3]} | {b[4]} | {b[5]}")
        print("---+---+---")
        print(f" {b[6]} | {b[7]} | {b[8]}")

if __name__ == "__main__":
    env = TicTacToeEnv()
    env.render()
    print("Available actions:", env.available_actions())

    print("\nSimulating moves...")
    state, reward, done, winner = env.step(0, 'X')
    env.render()
    print(f"Reward: {reward}, Done: {done}, Winner: {winner}")
