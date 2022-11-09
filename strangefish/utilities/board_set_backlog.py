"""
Copyright © 2022 The Johns Hopkins University Applied Physics Laboratory LLC

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
"""

from dataclasses import dataclass
from typing import List

import chess

from strangefish.utilities import populate_next_board_set


@dataclass
class OneTurnInfo:
    has_op_capture_square = False
    op_capture_square = None
    has_sense_result = False
    sense_result = None
    has_move_result = False
    move_result = None


class StoredOldBoardSet:
    """ Data structure for maintaining and updating board state hypotheses backlog """
    def __init__(self, boards: List[chess.Board]):
        self.boards: List[chess.Board] = boards
        self.turns_since_this_state: List[OneTurnInfo] = []
        self.expected_ratio = 1

    def add_info(self, info_kind, info_details, observed_ratio):
        if info_kind == "op_capture":
            self.turns_since_this_state.append(OneTurnInfo())
            self.turns_since_this_state[-1].has_op_capture_square = True
            self.turns_since_this_state[-1].op_capture_square = info_details
        elif info_kind == "sense_result":
            self.turns_since_this_state[-1].has_sense_result = True
            self.turns_since_this_state[-1].sense_result = info_details
        elif info_kind == "move_result":
            self.turns_since_this_state[-1].has_move_result = True
            self.turns_since_this_state[-1].move_result = info_details
        else:
            raise ValueError(f"Unexpected update action label: {info_kind}")
        self.expected_ratio *= observed_ratio

    @property
    def expected_size(self):
        return len(self.boards) * self.expected_ratio

    @property
    def is_empty(self):
        return not self.boards

    def expand_one_board(self):
        stored_board_set = {self.boards.pop()}
        priority_boards = set()
        for turn_info in self.turns_since_this_state:
            next_turn_boards, priority_boards = populate_next_board_set(
                stored_board_set,
                required_op_capture_square=turn_info.op_capture_square,
                required_sense_result=None if not turn_info.has_sense_result else turn_info.sense_result,
                required_move_result=None if not turn_info.has_move_result else turn_info.move_result,
                rc_disable_pbar=True,
            )
            stored_board_set = next_turn_boards[turn_info.op_capture_square]

            if not stored_board_set:
                return stored_board_set, priority_boards

        return stored_board_set, priority_boards


class BoardSetBacklog:
    """ Collection of one or more StoredOldBoardSet """
    def __init__(self):
        self.backlogs: List[StoredOldBoardSet] = []

    def add_row(self, boards):
        self.backlogs.append(StoredOldBoardSet(boards))

    def add_info(self, info_kind, info_details, observed_ratio):
        for row in self.backlogs:
            row.add_info(info_kind, info_details, observed_ratio)

    @property
    def is_empty(self):
        return not self.backlogs

    @property
    def total_stored_boards(self):
        return sum(len(row.boards) for row in self.backlogs)

    @property
    def stored_boards_and_turns_since(self):
        return tuple(
            (len(row.boards), len(row.turns_since_this_state))
            for row in self.backlogs
        )

    @property
    def expected_size(self):
        return sum(row.expected_size for row in self.backlogs)

    def expand_one_old_board(self):
        most_recent_row = self.backlogs[-1]
        new_board_set, priority_boards_in_set = most_recent_row.expand_one_board()
        if most_recent_row.is_empty:
            self.backlogs.pop()
        return new_board_set, priority_boards_in_set
