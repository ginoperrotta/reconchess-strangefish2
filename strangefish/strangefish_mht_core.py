"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
"""

import logging
import os
from collections import defaultdict
from functools import partial
from time import time, sleep
from tqdm import tqdm
from typing import Optional, List, Tuple, Set, Dict

import chess.engine
from reconchess import Player, Color, GameHistory, WinReason, Square

from strangefish.utilities import board_matches_sense, update_board_by_move, populate_next_board_set, PASS
from strangefish.utilities.board_set_backlog import BoardSetBacklog
from strangefish.utilities.player_logging import create_file_handler, create_stream_handler
from strangefish.utilities.timing import Timer

# Parameters for minor bot behaviors
RC_DISABLE_PBAR = os.getenv('RC_DISABLE_PBAR', 'false').lower() == 'true'  # Flag to disable the tqdm progress bars
WAIT_LOOP_RATE_LIMIT = 2  # minimum seconds spent looping in self.while_we_wait()

# Parameters for switching to the emergency backup plan
BOARD_SET_LIMIT = 1_000  # number of boards in set at which we stop processing and store for later
REPOPULATE_BOARD_SET_TARGET = 30  # number of boards at which we stop repopulating from stored states


class StrangeFish(Player):
    """
    StrangeFish is the main skeleton of our reconchess-playing bot. Its primary role is to manage the set of all
    possible board states based on the given information. Decision making for sense and move choices are left to be
    implemented in subclasses.

    RBC agents written as subclasses of StrangeFish must implement `sense_strategy(self, sense_actions: List[Square],
    move_actions: List[chess.Move], seconds_left: float) -> Optional[Square]` and `move_strategy(self, move_actions:
    List[chess.Move], seconds_left: float) -> Optional[chess.Move]` which actually make the gameplay decisions. Most
    subclass agents will also want to add code to `downtime_strategy` to make use of time on the opponent's turn.
    """

    def __init__(
        self,
        log_to_file=True,
        stream_log_level=logging.INFO,
        game_id=None,
        rc_disable_pbar=RC_DISABLE_PBAR,
    ):
        """
        Set up the MHT core of StrangeFish.

        :param log_to_file: A boolean flag to turn on/off logging to file game_logs/game_<game_id>.log
        :param stream_log_level: Logging level of detail to print to the screen
        :param game_id: Any printable identifier for logging (typically, the game number given by the server)
        :param rc_disable_pbar: A boolean flag to turn on/off the tqdm progress bars
        """

        self.boards: Set[chess.Board] = set()
        self.board_sample_priority: Dict[int, Set[chess.Board]] = {}
        self.next_turn_boards: defaultdict[Optional[chess.Square], Set[chess.Board]] = defaultdict(set)
        self.next_turn_boards_unsorted: Set[chess.Board] = set()
        self.num_boards_at_end_of_turn = None
        self.stored_old_boards = None

        self.color = None
        self.turn_num = None
        self.projected_end_time = None

        self.rc_disable_pbar = rc_disable_pbar

        game_log = logging.getLogger(f"game-{game_id}")
        game_log.setLevel(logging.DEBUG)
        game_log.addHandler(create_stream_handler(stream_log_level))
        if log_to_file:
            game_log.addHandler(create_file_handler(f"game_{game_id}.log"))

        self.logger = logging.getLogger(f"game-{game_id}.agent")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("A new StrangeFish player was initialized.")

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        color_name = chess.COLOR_NAMES[color]

        self.logger.info(f"Starting a new game as {color_name} against {opponent_name}.")
        self.boards = {board}
        self.board_sample_priority = defaultdict(set)
        self.stored_old_boards = BoardSetBacklog()
        self.color = color
        self.turn_num = 0
        self.num_boards_at_end_of_turn = 1
        self.projected_end_time = time() + 900

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.turn_num += 1
        self.logger.debug("Starting turn %d.", self.turn_num)

        # Do not "handle_opponent_move_result" if no one has moved yet
        if self.turn_num == 1 and self.color == chess.WHITE:
            return

        if captured_my_piece:
            self.logger.debug(f"Opponent captured my piece at {chess.SQUARE_NAMES[capture_square]}.")
        else:
            self.logger.debug("Opponent's move was not a capture.")

        # If creation of new board set didn't complete during op's turn (self.boards will not be empty)
        while self.boards and len(self.next_turn_boards[capture_square]) < BOARD_SET_LIMIT:
            self.expand_one_board(required_op_capture_square=capture_square)

        if self.boards:
            self.logger.debug(f"Board set has exceeded limit, storing {len(self.boards)} to process later.")
            # will still have boards in set if expanded already to limit
            self.stored_old_boards.add_row(self.boards)
        num_boards_expanded = self.num_boards_at_end_of_turn - len(self.boards)
        observed_ratio = max(1., len(self.next_turn_boards[capture_square])) / max(1., num_boards_expanded)
        self.stored_old_boards.add_info("op_capture", capture_square, observed_ratio)

        # Get this turn's board set from a dictionary keyed by the possible capture squares
        self.boards = self.next_turn_boards[capture_square]
        self.repopulate_exhausted_board_set()

        self.logger.debug(
            "Finished expanding and filtering the set of possible board states. "
            f"There are {len(self.boards)} possible boards "
            f"at the start of our turn {self.turn_num}."
        )

    def choose_sense(
        self,
        sense_actions: List[Square],
        move_actions: List[chess.Move],
        seconds_left: float,
    ) -> Optional[Square]:

        self.logger.debug(
            f"Choosing a sensing square for turn {self.turn_num} "
            f"with {len(self.boards)} boards and {seconds_left:.0f} seconds remaining."
        )

        # The option to pass isn't included in the reconchess input
        move_actions += [PASS]

        with Timer(self.logger.debug, "choosing sense location"):
            # Pass the needed information to the decision-making function to choose a sense square
            sense_choice = self.sense_strategy(sense_actions, move_actions, seconds_left)

        self.logger.debug(f"Chose to sense {chess.SQUARE_NAMES[sense_choice] if sense_choice else 'nowhere'}")

        return sense_choice

    def sense_strategy(
        self,
        sense_actions: List[Square],
        move_actions: List[chess.Move],
        seconds_left: float,
    ) -> Optional[Square]:
        raise NotImplementedError

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):

        # Filter the possible board set to only boards which would have produced the observed sense result
        num_before = len(self.boards)
        i = tqdm(
            self.boards,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} Filtering {len(self.boards)} boards by sense results",
            unit="boards",
        )
        self.boards = {
            board for board in map(partial(board_matches_sense, sense_result=sense_result), i) if board is not None
        }
        self.logger.debug(f"There were {num_before} possible boards before sensing " f"and {len(self.boards)} after.")

        observed_ratio = max(1., len(self.boards)) / num_before
        self.stored_old_boards.add_info("sense_result", sense_result, observed_ratio)
        self.repopulate_exhausted_board_set()

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.projected_end_time = time() + seconds_left

        # Currently, move_actions is passed by reference, so if we add the null move here it will be in the list twice
        #  since we added it in choose_sense also. Instead of removing this line altogether, I'm leaving a check so we
        #  are prepared in the case that reconchess is updated to pass a copy of the move_actions list instead.
        if PASS not in move_actions:
            move_actions += [PASS]

        self.logger.debug(
            f"Choosing move for turn {self.turn_num} "
            f"from {len(move_actions)} moves over {len(self.boards)} boards "
            f"with {seconds_left:.2f} seconds remaining."
        )

        with Timer(self.logger.debug, "choosing move"):
            # Pass the needed information to the decision-making function to choose a move
            move_choice = self.move_strategy(move_actions, seconds_left)

        self.logger.debug(f"The chosen move was {move_choice}")

        # reconchess uses None for the null move, so correct the function output if that was our choice
        return move_choice if move_choice != PASS else None

    def move_strategy(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        raise NotImplementedError

    def handle_move_result(
        self,
        requested_move: Optional[chess.Move],
        taken_move: Optional[chess.Move],
        captured_opponent_piece: bool,
        capture_square: Optional[Square],
    ):

        self.logger.debug(f"The requested move was {requested_move} and the taken move was {taken_move}.")
        if captured_opponent_piece:
            self.logger.debug(f"Move {taken_move} was a capture!")

        num_boards_before_filtering = len(self.boards)

        if requested_move is None:
            requested_move = PASS
        if taken_move is None:
            taken_move = PASS

        # Filter the possible board set to only boards on which the requested move would have resulted in the taken move
        i = tqdm(
            self.boards,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} Filtering {len(self.boards)} boards by move results",
            unit="boards",
        )
        self.boards = {
            board
            for board in map(
                partial(
                    update_board_by_move,
                    requested_move=requested_move,
                    taken_move=taken_move,
                    captured_opponent_piece=captured_opponent_piece,
                    capture_square=capture_square,
                ),
                i,
            )
            if board is not None
        }

        self.logger.debug(
            f"There were {num_boards_before_filtering} possible boards "
            f"before filtering and {len(self.boards)} after."
        )

        observed_ratio = max(1., len(self.boards)) / num_boards_before_filtering
        self.stored_old_boards.add_info("move_result", (requested_move, taken_move, capture_square), observed_ratio)

        # Re-initialize the set of boards for next turn (filled in while_we_wait and/or handle_opponent_move_result)
        self.num_boards_at_end_of_turn = len(self.boards)
        self.next_turn_boards = defaultdict(set)
        self.next_turn_boards_unsorted = set()
        self.board_sample_priority = defaultdict(set)

    def expand_one_board(
            self,
            required_op_capture_square: Optional[chess.Square] = False,
    ):
        if not self.boards:
            return
        new_board_set, board_sample_priority = populate_next_board_set(
            {self.boards.pop()},
            required_op_capture_square=required_op_capture_square,
            rc_disable_pbar=True,
        )
        for priority, boards in board_sample_priority.items():
            self.board_sample_priority[priority] |= boards
        for square, boards in new_board_set.items():
            self.next_turn_boards[square] |= boards
            self.next_turn_boards_unsorted |= boards

    def expand_one_old_board(self):
        if self.stored_old_boards.is_empty:
            return 0
        new_boards, board_sample_priority = self.stored_old_boards.expand_one_old_board()
        self.boards |= new_boards
        for priority, boards in board_sample_priority.items():
            self.board_sample_priority[priority] |= boards
        return len(new_boards)

    def repopulate_exhausted_board_set(self):
        if self.stored_old_boards.is_empty:
            return
        for _boards, _steps in self.stored_old_boards.stored_boards_and_turns_since:
            self.logger.debug(f"Currently have {_boards} boards stored from {_steps} turns ago.")
        self.logger.debug(f"Estimate {self.stored_old_boards.total_stored_boards} stored old boards"
                          f" would expand into {round(self.stored_old_boards.expected_size)} current boards.")
        if len(self.boards) >= REPOPULATE_BOARD_SET_TARGET:
            return
        original_set_size = len(self.boards)
        total = self.stored_old_boards.total_stored_boards
        self.logger.debug(f"Repopulating exhausted board set from {total} stored boards")
        with tqdm(total=total, desc="Repopulating exhausted board set", disable=self.rc_disable_pbar) as pbar:
            prev_total = total
            while not self.stored_old_boards.is_empty and len(self.boards) < REPOPULATE_BOARD_SET_TARGET:
                self.expand_one_old_board()
                new_total = self.stored_old_boards.total_stored_boards
                pbar.update(n=prev_total - new_total)
                prev_total = new_total
                if time() > self.projected_end_time:
                    self.logger.warning("Breaking board set construction due to time limit")
                    break
        self.logger.debug(f"Constructed {len(self.boards) - original_set_size} new boards from stored past states.")
        for _boards, _steps in self.stored_old_boards.stored_boards_and_turns_since:
            self.logger.debug(f"Currently have {_boards} boards stored from {_steps} turns ago.")

    def while_we_wait(self):
        start_time = time()
        self.logger.debug(
            "Running the `while_we_wait` method. "
            f"{len(self.boards)} boards left to expand for next turn."
        )

        logged_current_board_message = False
        logged_old_board_message = False

        while time() - start_time < WAIT_LOOP_RATE_LIMIT:

            # If there are still boards in the set from last turn, remove one and expand it by all possible moves
            if self.boards:
                if not logged_current_board_message:
                    logged_current_board_message = True
                    self.logger.debug("Expanding current boards for next turn's board set")
                self.expand_one_board()
            elif not self.stored_old_boards.is_empty:
                if not logged_old_board_message:
                    logged_old_board_message = True
                    self.logger.debug("Expanding stored old boards for next turn's board set")
                num_new_boards = self.expand_one_old_board()
                self.num_boards_at_end_of_turn += num_new_boards

            # If all of last turn's boards have been expanded, pass to the sense/move function's waiting method
            else:
                self.downtime_strategy()

    def downtime_strategy(self):
        sleep(0.1)

    def handle_game_end(
        self,
        winner_color: Optional[Color],
        win_reason: Optional[WinReason],
        game_history: GameHistory,
    ):
        self.logger.info(
            f"I {'won' if winner_color == self.color else 'lost'} "
            f"by {win_reason.name if hasattr(win_reason, 'name') else win_reason}"
        )
        self.gameover_strategy()

    def gameover_strategy(self):
        pass
