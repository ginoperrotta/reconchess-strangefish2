"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

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
from dataclasses import dataclass

import chess.engine

from strangefish.utilities import generate_rbc_moves, count_set_bits, PIECE_VALUES


@dataclass
class ScoreConfig:
    capture_king_score: float = 5_000  # bonus points for a winning move
    checkmate_score: int = 4_000  # point value of checkmate
    points_per_move_to_mate: int = 0.95  # points ratio for each turn away from checkmate
    into_check_score: float = -6_000  # point penalty for moving into check
    op_into_check_score: float = -3_000  # different score for opponent moving into check
    search_depth: int = 8  # Stockfish engine search ply
    reward_attacker: float = 25  # Bonus points if move sets up attack on enemy king
    require_sneak: bool = False  # Only reward bonus points to aggressive moves if they are sneaky (aren't captures)
    unsafe_attacker_penalty: float = -20
    search_check_in_two: bool = True
    search_check_in_three: bool = False
    check_risk_penalty: float = -25
    capture_penalty: float = -15
    passing_reward: float = 15
    use_absolute_score: float = 0.75
    contempt: int = 0
    reward_num_moves_ratio: float = 0.2
    weak_squares_score: float = 0.5
    op_num_moves_ratio: float = 0.1
    skip_bonus_evals: bool = False


def calculate_score(
    engine: chess.engine.SimpleEngine,
    board,
    move=chess.Move.null(),
    prev_turn_score=0,
    is_op_turn=False,
    score_config: ScoreConfig = ScoreConfig(),
):

    pov = board.turn

    if board.king(not pov) is None:  # This shouldn't be possible, just checking and logging in case of bugs
        logging.getLogger().warning(f"Requested score when their king is missing! {board.fen()}: {move.uci()}")
        return (
            score_config.op_into_check_score
            if is_op_turn
            else score_config.into_check_score + prev_turn_score * score_config.use_absolute_score
        )
    if board.king(pov) is None:  # This shouldn't be possible, just checking and logging in case of bugs
        logging.getLogger().warning(f"Requested score when our own king is missing! {board.fen()}: {move.uci()}")
        return (
            score_config.op_into_check_score
            if is_op_turn
            else score_config.into_check_score + prev_turn_score * score_config.use_absolute_score
        )

    next_board = board.copy(stack=False)
    next_board.push(move)
    next_board.clear_stack()

    if next_board.king(not pov) is None:
        return score_config.capture_king_score + prev_turn_score * score_config.use_absolute_score
    if next_board.king(pov) is None:  # This shouldn't be possible, just checking and logging in case of bugs
        logging.getLogger().warning(
            f"Requested score when our own king is missing (after the move)! {board.fen()}: {move.uci()}"
        )
        return (
            score_config.op_into_check_score
            if is_op_turn
            else score_config.into_check_score + prev_turn_score * score_config.use_absolute_score
        )

    next_turn_board = next_board.copy(stack=False)
    next_turn_board.push(chess.Move.null())
    next_turn_board.clear_stack()

    if next_board.was_into_check():
        return (
            score_config.op_into_check_score
            if is_op_turn
            else score_config.into_check_score + prev_turn_score * score_config.use_absolute_score
        )

    try:
        engine_result = engine.analyse(
            next_board,
            chess.engine.Limit(depth=score_config.search_depth),
            options={"contempt": score_config.contempt},
        )["score"].pov(pov)

        if type(engine_result) == chess.engine.Cp:  # if the result is a centipawn rating
            score = engine_result.score()
        else:  # otherwise it must be checkmate
            score = engine_result.score(
                mate_score=score_config.checkmate_score
            ) * score_config.points_per_move_to_mate ** abs(engine_result.score(mate_score=0))

    except chess.engine.EngineTerminatedError:
        score = score_material(next_board, pov)

    if score_config.skip_bonus_evals:
        return score - prev_turn_score * (1 - score_config.use_absolute_score)

    # penalize captures to reduce position-revealing moves
    if board.is_capture(move):
        score += score_config.capture_penalty

    # reward passing to increase uncertainty
    if move == chess.Move.null() and not is_op_turn:
        score += score_config.passing_reward

    # penalize leaving open attack squares to our king
    score -= count_weak_squares(next_turn_board) * score_config.weak_squares_score

    op_moves = list(generate_rbc_moves(next_board))
    next_moves = list(generate_rbc_moves(next_turn_board))

    # Add bonus board position score if king is attacked
    king_attackers = next_board.attackers(pov, next_board.king(not pov))  # list of pieces that can reach the enemy king
    if king_attackers:  # if there are any such pieces...
        # and we don't require the attackers to be sneaky
        # or if we do require the attackers to be sneaky, either the last move was not a capture (which would give away
        # our position) or there are now attackers other than the piece that moves (discovered check)
        if (
            not score_config.require_sneak
            or not board.is_capture(move)
            or any(square != move.to_square for square in king_attackers)
        ):
            score += score_config.reward_attacker  # add the bonus points
            # but penalize if none of the attackers is on a safe square
            if not is_op_turn and all(next_board.attackers(not pov, square) for square in king_attackers):
                score += score_config.unsafe_attacker_penalty
    elif score_config.search_check_in_two and check_in_two(next_turn_board, next_moves, score_config):
        score += score_config.reward_attacker / 2
        if not is_op_turn and move != chess.Move.null() and next_board.attackers(not pov, move.to_square):
            score += score_config.unsafe_attacker_penalty / 2
    elif score_config.search_check_in_three and check_in_three(next_turn_board, next_moves, score_config):
        score += score_config.reward_attacker / 4
        if not is_op_turn and move != chess.Move.null() and next_board.attackers(not pov, move.to_square):
            score += score_config.unsafe_attacker_penalty / 4

    # Also look to see if any op move could put us in check
    if not is_op_turn and search_op_check(next_board, op_moves):
        score += score_config.check_risk_penalty

    score += len(next_moves) * score_config.reward_num_moves_ratio
    score -= len(op_moves) * score_config.op_num_moves_ratio

    score -= prev_turn_score * (1 - score_config.use_absolute_score)

    return score


def search_op_check(board, moves):
    for next_move in moves:
        if next_move != chess.Move.null():
            if not board.is_capture(next_move):
                if board.gives_check(next_move):
                    return True
    return False


def check_in_two(board, moves, score_config):
    for next_move in moves:
        if next_move != chess.Move.null():
            if not score_config.require_sneak or not board.is_capture(next_move):
                if board.gives_check(next_move):
                    if score_config.unsafe_attacker_penalty == 0 or not board.attackers(
                        not board.turn, next_move.to_square
                    ):
                        return True
    return False


def check_in_three(board, moves, score_config):
    for next_move in moves:
        if next_move != chess.Move.null():
            if not score_config.require_sneak or not board.is_capture(next_move):
                board.push(next_move)
                board.push(chess.Move.null())
                for next_next_move in generate_rbc_moves(board):
                    if next_next_move != chess.Move.null():
                        if not score_config.require_sneak or not board.is_capture(next_next_move):
                            if board.gives_check(next_next_move):
                                if score_config.unsafe_attacker_penalty == 0 or not board.attackers(
                                    not board.turn, next_move.to_square
                                ):
                                    return True
                board.pop()
                board.pop()
    return False


def count_weak_squares(board: chess.Board):
    color = board.turn
    king = board.king(color)
    weak_squares = chess.BB_EMPTY
    my_pieces = board.occupied_co[color]
    op_pieces = board.occupied_co[not color]
    op_has_queen = bool(board.queens & op_pieces)
    op_has_rook = bool(board.rooks & op_pieces)
    op_has_bishop = bool(board.bishops & op_pieces)
    op_has_knight = bool(board.knights & op_pieces)

    if op_has_queen or op_has_rook:
        weak_squares |= chess.BB_RANK_ATTACKS[king][chess.BB_RANK_MASKS[king] & my_pieces]
        weak_squares |= chess.BB_FILE_ATTACKS[king][chess.BB_FILE_MASKS[king] & my_pieces]
    if op_has_queen or op_has_bishop:
        weak_squares |= chess.BB_DIAG_ATTACKS[king][chess.BB_DIAG_MASKS[king] & my_pieces]
    if op_has_knight:
        weak_squares |= chess.BB_KNIGHT_ATTACKS[king]

    weak_squares &= ~my_pieces

    return bin(weak_squares).count("1")


def score_material(board: chess.Board, color: chess.Color):
    return sum(
        (count_set_bits(board.pieces_mask(piece, color)) - count_set_bits(board.pieces_mask(piece, not color)))
        * PIECE_VALUES[piece]
        for piece in PIECE_VALUES
    )
