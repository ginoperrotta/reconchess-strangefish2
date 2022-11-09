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

from typing import Iterable, Optional, Set, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import signal

import chess
from reconchess.utilities import (
    without_opponent_pieces,
    is_illegal_castle,
    is_psuedo_legal_castle,
    moves_without_opponent_pieces,
    pawn_capture_moves_on,
    capture_square_of_move,
)

# These are the possible squares to search–all squares that aren't on the edge of the board.
SEARCH_SPOTS = [
    9, 10, 11, 12, 13, 14,
    17, 18, 19, 20, 21, 22,
    25, 26, 27, 28, 29, 30,
    33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46,
    49, 50, 51, 52, 53, 54,
]

# Centipawn scores for pieces
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.ROOK: 563,
    chess.KNIGHT: 305,
    chess.BISHOP: 333,
    chess.QUEEN: 950,
    # chess.KING: 0
}


# Instantiate pass move for more efficient frequent comparison
PASS = chess.Move.null()


# Add a hash method for chess.Board objects so that they can be tested for uniqueness, and
# modify the equality method to match. (Neither now uses turn counters, which don't matter in RBC)
chess.Board.__hash__ = lambda self: hash(self._transposition_key())
chess.Board.__eq__ = lambda self, board: self._transposition_key() == board._transposition_key()


# Generate all RBC-legal moves for a board
def rbc_legal_moves(
        board: chess.Board,
        capture_mask: Optional[chess.Bitboard] = None,
) -> List[chess.Move]:

    if capture_mask is not None:
        # Adding behavior here to compute only moves corresponding to observed captures
        moves = list(board.generate_pseudo_legal_moves(to_mask=capture_mask))
        if board.ep_square is not None:
            ep_capture_square = board.ep_square + (8 if board.turn == chess.BLACK else -8)
            if chess.BB_SQUARES[ep_capture_square] & capture_mask:
                moves += list(board.generate_pseudo_legal_ep())
        return moves

    moves = [PASS]
    moves += list(board.generate_pseudo_legal_moves())
    # TODO: can we just hard code the castling moves here? No need to run without_opponent_pieces that way
    moves += [move for move in without_opponent_pieces(board).generate_castling_moves()
              if not is_illegal_castle(board, move)]
    return moves


# Generate all possible moves from just our own pieces
def rbc_legal_move_requests(board: chess.Board) -> Iterable[chess.Move]:
    return (
        moves_without_opponent_pieces(board)
        + pawn_capture_moves_on(board)
        + [PASS]
    )


# Produce a sense result from a hypothetical true board and a sense square
def simulate_sense(board, square):  # copied (with modifications) from LocalGame
    if square is None:
        # don't sense anything
        sense_result = []
    else:
        if square not in list(chess.SQUARES):
            raise ValueError(f"{square} is not a valid square.")
        rank, file = chess.square_rank(square), chess.square_file(square)
        sense_result = []
        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                if 0 <= rank + delta_rank <= 7 and 0 <= file + delta_file <= 7:
                    sense_square = chess.square(file + delta_file, rank + delta_rank)
                    sense_result.append((sense_square, board.piece_at(sense_square)))
    return tuple(sense_result)


# Collect the sensed area squares for each sense choice center square
def generate_sense_areas():
    sense_area = {square: [] for square in chess.SQUARES}
    sense_mask = {square: chess.BB_EMPTY for square in chess.SQUARES}
    for square in chess.SQUARES:
        rank, file = chess.square_rank(square), chess.square_file(square)
        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                if 0 <= rank + delta_rank <= 7 and 0 <= file + delta_file <= 7:
                    sense_square = chess.square(file + delta_file, rank + delta_rank)
                    sense_area[square].append(sense_square)
                    sense_mask[square] |= chess.BB_SQUARES[sense_square]
    sense_area[None] = []
    sense_mask[None] = chess.BB_EMPTY
    return sense_area, sense_mask


SENSE_AREAS, SENSE_MASKS = generate_sense_areas()
SENSE_MASK_LIST = tuple(SENSE_MASKS[sq] for sq in chess.SQUARES)


# Faster, less readable sense result from a hypothetical true board and a sense square
def sense_masked_bitboards(board: chess.Board, square: chess.Square):
    mask = SENSE_MASK_LIST[square]
    return tuple(
        bitboard & mask
        for bitboard in (
            *board.occupied_co,
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
        )
    )


# test an attempted move on a board to see what move is actually taken
def simulate_move(board, move, pseudo_legal_moves=None):
    if move == PASS:
        return None
    if pseudo_legal_moves is None:
        pseudo_legal_moves = list(board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[move.from_square]))
    # if its a legal move, don't change it at all (generate_pseudo_legal_moves does not include pseudo legal castles)
    if move in pseudo_legal_moves or is_psuedo_legal_castle(board, move):
        return move
    if is_illegal_castle(board, move):
        return None
    # if the piece is a sliding piece, slide it as far as it can go
    piece = board.piece_at(move.from_square)
    if piece.piece_type in [chess.PAWN, chess.ROOK, chess.BISHOP, chess.QUEEN]:
        move = slide_move(move, pseudo_legal_moves)
    return move


def slide_move(move: chess.Move, pseudo_legal_moves: Iterable[chess.Move]) -> Optional[chess.Move]:
    # Copied from reconchess.utilities to prevent redundant generation of pseudo-legal moves
    squares = chess.SquareSet(chess.between(move.from_square, move.to_square))
    if move.to_square > move.from_square:
        squares = reversed(squares)
    for slide_square in [move.to_square] + list(squares):
        revised = chess.Move(move.from_square, slide_square, move.promotion)
        if revised in pseudo_legal_moves:
            return revised
    return None


# check if a taken move would have happened on a board
def update_board_by_move(
    board: chess.Board,
    requested_move: Optional[chess.Move],
    taken_move: Optional[chess.Move],
    captured_opponent_piece: bool,
    capture_square: Optional[chess.Square],
) -> Optional[chess.Board]:
    # if the taken move was a capture...
    if captured_opponent_piece:
        # the board is invalid if the capture would not have happened
        if not board.is_capture(taken_move):
            return None
        # the board is invalid if the captured piece would have been the king
        # (wrong if it really was the king, but then the game is over)
        captured_piece = board.piece_at(capture_square)
        if captured_piece and captured_piece.piece_type == chess.KING:
            return None
        # ensure that the capture squares match (for en passant)
        if capture_square != capture_square_of_move(board, taken_move):
            return None
    # if the taken move was not a capture...
    elif taken_move != PASS:
        # the board is invalid if a capture would have happened
        if board.is_capture(taken_move):
            return None
    # invalid if the requested move would have not resulted in the taken move
    if (simulate_move(board, requested_move) or PASS) != taken_move:
        return None
    # otherwise the board is still valid
    next_board = fast_copy_board(board)
    next_board.push(taken_move)
    return next_board


# Expand one turn's boards into next turn's set by all possible moves. Store as dictionary keyed by capture square.
def populate_next_board_set(
        board_set: Set[chess.Board],
        required_op_capture_square=False,
        required_sense_result=None,
        required_move_result=None,
        rc_disable_pbar: bool = False
):
    next_turn_boards = defaultdict(set)
    board_sample_priority = defaultdict(set)
    for board in tqdm(board_set, disable=rc_disable_pbar, unit="boards",
                      desc=f"Expanding {len(board_set)} boards into new set"):
        for next_board, capture_square, priority in get_next_boards_and_capture_squares(
            board,
            required_op_capture_square=required_op_capture_square,
            required_sense_result=required_sense_result,
            required_move_result=required_move_result,
        ):
            next_turn_boards[capture_square].add(next_board)
            board_sample_priority[priority].add(next_board)
    return next_turn_boards, board_sample_priority


# Check if a board could have produced these sense results
def board_matches_sense(board, sense_result):
    for square, piece in sense_result:
        if board.piece_at(square) != piece:
            return None
    return board


def sense_partition_leq(partition_a, partition_b):
    # Determines if the board set division of partition_a is at least as informative as partition_b
    # (leq = less than or equal which for a set partition indicates a finer division)
    return all(any(subset_a.issubset(subset_b) for subset_b in partition_b) for subset_a in partition_a)


# Generate tuples of next turn's boards and capture squares for one current board
def get_next_boards_and_capture_squares(
        board: chess.Board,
        required_op_capture_square=False,  # False so that the None option remains
        required_sense_result=None,
        required_move_result=None,
):
    # Calculate all possible opponent moves from this board state
    starts_in_check = board.is_check()
    results = []
    if required_op_capture_square is None or required_op_capture_square is False:
        capture_mask = None
    else:
        capture_mask = chess.BB_SQUARES[required_op_capture_square]
    for move in rbc_legal_moves(board, capture_mask=capture_mask):
        priority = 0
        op_capture_square = capture_square_of_move(board, move)
        if required_op_capture_square is not False and op_capture_square != required_op_capture_square:
            continue
        if op_capture_square is not None:
            captured_piece = board.piece_at(op_capture_square)
            if captured_piece and captured_piece.piece_type == chess.KING:
                continue
        board.push(move)
        if required_sense_result is None or board_matches_sense(board, required_sense_result):
            # Returned results will be screened by this sense result *before* and board copies are made
            if required_move_result is None:
                next_board = fast_copy_board(board)
                priority = assign_priority(next_board, starts_in_check)
                results.append((next_board, op_capture_square, priority))
            else:
                # Returned results will be screened by this move result *before* and board copies are made
                # Returned results will already have the player's own move pushed as well
                requested_move, taken_move, capture_square = required_move_result
                captured_opponent_piece = capture_square is not None
                next_board = update_board_by_move(board, requested_move, taken_move, captured_opponent_piece, capture_square)
                if next_board is not None:
                    results.append((next_board, op_capture_square, priority))
        board.pop()
    return results


# Change any promotion moves to choose queen
def force_promotion_to_queen(move: chess.Move):
    if move.promotion not in (None, chess.QUEEN):
        move.promotion = chess.QUEEN
    return move


def ignore_one_term(signum, frame):  # Let a sub-process survive the first ctrl-c call for graceful game exiting
    # reset to default response to interrupt signals
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def count_set_bits(x: int):
    return bin(x).count("1")


def fast_copy_board(board: chess.Board) -> chess.Board:
    # chess.Board.__init__ is wastefully repetitive for copies
    #   Instead, manually perform the combined init+copy
    new_board = object.__new__(chess.Board)

    new_board.pawns = board.pawns
    new_board.knights = board.knights
    new_board.bishops = board.bishops
    new_board.rooks = board.rooks
    new_board.queens = board.queens
    new_board.kings = board.kings

    new_board.occupied_co = [*board.occupied_co]
    new_board.occupied = board.occupied
    new_board.promoted = board.promoted

    new_board.chess960 = board.chess960

    new_board.ep_square = board.ep_square
    new_board.castling_rights = board.castling_rights
    new_board.turn = board.turn
    new_board.fullmove_number = board.fullmove_number
    new_board.halfmove_clock = board.halfmove_clock

    new_board.move_stack = []
    new_board._stack = []

    return new_board


def print_sense_result(sense_result: Iterable[Tuple[chess.Square, Optional[chess.Piece]]]):
    return "".join("-" if piece is None else piece.symbol() for square, piece in sense_result)


def could_move_into_check(board: chess.Board):
    king_square = board.king(board.turn)
    # First check king moves
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square] & ~board.occupied_co[board.turn]):
        if board.is_attacked_by(not board.turn, square):
            return True
    # Copying methods from chess.Board.is_into_check to move redundant computations out of this loop
    blockers = board._slider_blockers(king_square)
    for move in board.generate_pseudo_legal_moves(from_mask=blockers):
        if not board._is_safe(king_square, blockers, move):
            return True
    return False


def assign_priority(board: chess.Board, starts_in_check: bool):
    if starts_in_check and not board.was_into_check():  # Opponent evaded check
        priority = 5
    elif board.is_check():  # If the player is in check
        if not any(
            board.is_capture(move)
            for move in board._generate_evasions(
                king=board.king(board.turn),
                checkers=board.checkers_mask(),
                from_mask=chess.BB_SQUARES[board.king(board.turn)],
            )
        ):  # Test: remove priority if attacker can be captured as an evasion
            priority = 4
        else:
            priority = 3
    elif could_move_into_check(board):  # Sample boards in which I could move into check
        priority = 2
    elif any(
        board.is_attacked_by(not board.turn, square)
        for square in chess.SquareSet(board.occupied_co[board.turn] & board.queens)
    ):  # Also prioritize attacks on players queens
        priority = 1
    else:
        priority = 0
    return priority
