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

from dataclasses import dataclass
from functools import reduce
from operator import ior
import math

import chess.engine
from reconchess.utilities import capture_square_of_move

from strangefish.utilities import count_set_bits, PIECE_VALUES, PASS, fast_copy_board


ENGINE_SCORE_CACHE = {}
ENGINE_CACHE_STATS = [0, 0]  # hits, misses


@dataclass
class ScoreConfig:
    capture_king_score: float = 9_000  # bonus points for a winning move
    checkmate_score: int = 8_000  # point value of checkmate
    points_per_move_to_mate: int = 0.95  # points ratio for each turn away from checkmate
    into_check_score: float = -7_000  # point penalty for moving into check
    remain_in_check_penalty: float = -2_000  # additional point penalty for staying in check
    op_into_check_score: float = -4_000  # different score for opponent moving into check
    search_depth: int = 7  # Stockfish engine search ply
    reward_attacker: float = 20  # Bonus points if move sets up attack on enemy king
    reward_check_threats: float = 15  # Bonus points per next quiet move that could be a check
    require_sneak: bool = True  # Only reward bonus points to aggressive moves if they are sneaky (aren't captures)
    unsafe_attacker_penalty: float = -15
    hanging_material_penalty_ratio: float = 0.1
    capture_penalty: float = -5
    passing_reward: float = 5
    safe_pawn_pressure_reward: float = 15
    unsafe_pawn_pressure_reward: float = 5
    passed_pawn_reward: float = 15
    castling_rights_bonus: float = 15
    use_absolute_score: float = 0.85
    weak_squares_score: float = 1.0
    skip_bonus_evals: bool = False
    reduce_heuristic_when_equal: bool = False


def calculate_score(engine: chess.engine.SimpleEngine,
                    board, move=PASS,
                    prev_turn_score=0, is_op_turn=False,
                    score_config: ScoreConfig = ScoreConfig()):

    my_color = board.turn
    pov = not my_color if is_op_turn else my_color

    next_board = fast_copy_board(board)
    if not is_op_turn:
        next_board.push(move)
        next_board.clear_stack()
    else:
        assert move is None or move == PASS

    if next_board.king(not pov) is None:
        return score_config.capture_king_score \
               + score_material(next_board, pov) \
               + prev_turn_score * score_config.use_absolute_score

    if next_board.was_into_check():
        if is_op_turn:
            return score_config.op_into_check_score
        else:
            score = score_config.into_check_score + prev_turn_score * score_config.use_absolute_score
            if board.is_check():
                score += score_config.remain_in_check_penalty
            return score

    cache_key = next_board._transposition_key()
    try:
        score = ENGINE_SCORE_CACHE[cache_key]
    except KeyError:
        # For Stockfish versions based on NNUE, seem to need to screen illegal en passant
        replaced_ep_square = None
        if next_board.ep_square is not None and not next_board.has_legal_en_passant():
            replaced_ep_square, next_board.ep_square = next_board.ep_square, replaced_ep_square

        engine_result = engine.analyse(
            next_board, chess.engine.Limit(depth=score_config.search_depth), info=chess.engine.INFO_SCORE,
        )['score'].pov(pov)
        if type(engine_result) == chess.engine.Cp:  # if the result is a centipawn rating
            score = engine_result.score()
        else:  # otherwise it must be checkmate
            score = engine_result.score(mate_score=score_config.checkmate_score) * \
                score_config.points_per_move_to_mate ** abs(engine_result.score(mate_score=0)) + \
                score_material(next_board, pov)

        if replaced_ep_square is not None:
            next_board.ep_square = replaced_ep_square

        # Add to cache
        ENGINE_SCORE_CACHE[cache_key] = score
        ENGINE_CACHE_STATS[1] += 1
    else:
        ENGINE_CACHE_STATS[0] += 1

    if score_config.skip_bonus_evals:
        return score - prev_turn_score * (1 - score_config.use_absolute_score)

    if not is_op_turn and score_config.reduce_heuristic_when_equal:
        scale = abs(math.tanh(prev_turn_score / 500))
    else:
        scale = 1.0

    # penalize captures to reduce position-revealing moves
    if board.is_capture(move):
        score += score_config.capture_penalty * scale

    # reward passing to increase uncertainty
    if move == PASS and not is_op_turn:
        score += score_config.passing_reward * scale

    # penalize leaving open attack squares to our king
    score -= count_weak_squares(next_board, pov) * score_config.weak_squares_score * scale
    score += count_weak_squares(next_board, not pov) * score_config.weak_squares_score * scale

    # Check if move is a capture
    move_is_capture = board.is_capture(move)

    # Add bonus board position score if king is attacked
    op_king_square = next_board.king(not pov)
    king_attackers_mask = next_board.attackers_mask(pov, op_king_square)  # bitboard
    king_attackers = chess.SquareSet(king_attackers_mask)  # list of pieces that can reach the enemy king
    if king_attackers:  # if there are any such pieces...
        # and we don't require the attackers to be sneaky
        # or if we do require the attackers to be sneaky, either the last move was not a capture (which would give away
        # our position) or there are now attackers other than the piece that moves (discovered check)
        if not score_config.require_sneak or not move_is_capture or any(square != move.to_square for square in king_attackers):
            score += score_config.reward_attacker * scale  # add the bonus points
            # but penalize if any evasion if also a capture
            if not is_op_turn and any(
                    capture_square_of_move(next_board, op_move) is not None
                    for op_move in next_board._generate_evasions(op_king_square, king_attackers_mask)
            ):
                score += score_config.unsafe_attacker_penalty * scale

    # Reward quiet threats on king and hanging pieces, but penalize own hanging pieces

    # ... first for my pieces and threats
    num_checks, hanging_material = count_quiet_check_threats(next_board, my_color)
    score += score_config.reward_check_threats * num_checks * scale
    score -= score_config.hanging_material_penalty_ratio * hanging_material * scale

    # ... then inversely for my opponent's
    num_checks, hanging_material = count_quiet_check_threats(next_board, not my_color)
    score -= score_config.reward_check_threats * num_checks * scale
    score += score_config.hanging_material_penalty_ratio * hanging_material * scale
    # TODO: can this be combined into one call? does this take enough time to matter?

    if not is_op_turn:
        # Reward non-capture pawn moves that pressure opponent pieces
        score += reward_pawn_pressure(board, move, next_board, move_is_capture, pov, score_config) * scale

    # Reward passed pawns (not in the typical sense; here including pieces as blocking)
    if score_config.passed_pawn_reward:
        score += score_config.passed_pawn_reward * count_passed_pawns(next_board, pov) * scale
        score -= score_config.passed_pawn_reward * count_passed_pawns(next_board, not pov) * scale

    # Reward positions with castling rights maintained
    if score_config.castling_rights_bonus:
        score += score_config.castling_rights_bonus * next_board.has_castling_rights(pov) * scale
        score -= score_config.castling_rights_bonus * next_board.has_castling_rights(not pov) * scale

    score -= prev_turn_score * (1 - score_config.use_absolute_score)

    return score


def reward_pawn_pressure(board, move, next_board, move_is_capture, pov, score_config):
    # Reward non-capture pawn moves that pressure opponent pieces
    if move != PASS and not move_is_capture and board.piece_at(move.from_square).piece_type == chess.PAWN:
        attacks_mask = next_board.attacks_mask(move.to_square)
        op_material = (
            next_board.occupied_co[not pov]
            & (
                next_board.queens
                | next_board.rooks
                | next_board.bishops
                | next_board.knights
            )
        )
        pressured_material = count_set_bits(attacks_mask & op_material)
        if pressured_material:
            if next_board.is_attacked_by(pov, move.to_square):  # Pressuring pawn has support
                return score_config.safe_pawn_pressure_reward * pressured_material
            else:  # Unsupported attack
                return score_config.unsafe_pawn_pressure_reward * pressured_material

    # Otherwise
    return 0


def count_weak_squares(board: chess.Board, color: chess.Color):
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

    return count_set_bits(weak_squares)


PROMOTION_RISKS = [
    (chess.BB_RANK_2 | chess.BB_RANK_3),
    (chess.BB_RANK_7 | chess.BB_RANK_6),
]


def count_passed_pawns(board: chess.Board, color: chess.Color):
    pawn_squares = chess.SquareSet(board.pawns & board.occupied_co[color] & PROMOTION_RISKS[color])
    file_masks = [chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied] for square in pawn_squares]
    promotion_rank = chess.BB_RANK_8 if color is chess.WHITE else chess.BB_RANK_1
    passed_pawns = [bool(open_file & promotion_rank) for open_file in file_masks]
    return sum(passed_pawns)


def score_material(board: chess.Board, color: chess.Color):
    return sum(
        (count_set_bits(board.pieces_mask(piece, color))
         - count_set_bits(board.pieces_mask(piece, not color)))
        * value
        for piece, value in PIECE_VALUES.items()
    )


def count_quiet_check_threats(board: chess.Board, pov: chess.Color):
    # There are a lot of operations here, but almost all of them happen just to call board.is_check() so this is faster
    # This isn't really exhaustive, as pieces can block their own moves into check, but those moves are probably not
    # relevant to scoring anyways. Revealed checks are also not included.

    # Need to know the location of op king and all pieces that block ranks, files, and diagonals for that square
    op_king = board.king(not pov)  # This is a chess.Square, not a chess.BitBoard
    rank_pieces = chess.BB_RANK_MASKS[op_king] & board.occupied
    file_pieces = chess.BB_FILE_MASKS[op_king] & board.occupied
    diag_pieces = chess.BB_DIAG_MASKS[op_king] & board.occupied

    # Find squares that are either safe or supported
    op_pawns = board.pawns & board.occupied_co[not pov]
    op_pawns_attack = chess.BB_EMPTY if not op_pawns else \
        reduce(ior, [chess.BB_PAWN_ATTACKS[not pov][square] for square in chess.SquareSet(op_pawns)])
    op_knights = board.knights & board.occupied_co[not pov]
    op_knights_attack = chess.BB_EMPTY if not op_knights else \
        reduce(ior, [chess.BB_KNIGHT_ATTACKS[square] for square in chess.SquareSet(op_knights)])
    op_rooks = board.rooks & board.occupied_co[not pov]
    op_rooks_attack = chess.BB_EMPTY if not op_rooks else \
        reduce(ior, [chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & board.occupied] |
                     chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(op_rooks)])
    op_bishops = board.bishops & board.occupied_co[not pov]
    op_bishops_attack = chess.BB_EMPTY if not op_bishops else \
        reduce(ior, [chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(op_bishops)])
    op_queens = board.queens & board.occupied_co[not pov]
    op_queens_attack = chess.BB_EMPTY if not op_queens else \
        reduce(ior, [chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & board.occupied] |
                     chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied] |
                     chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(op_queens)])
    op_king_attacks = chess.BB_KING_ATTACKS[op_king]
    attacked_squares = (op_pawns_attack | op_knights_attack | op_rooks_attack |
                        op_bishops_attack | op_queens_attack | op_king_attacks)
    safe_squares = ~attacked_squares

    my_pawns = board.pawns & board.occupied_co[pov] & safe_squares
    my_pawns_attack = chess.BB_EMPTY if not my_pawns else \
        reduce(ior, [chess.BB_PAWN_ATTACKS[pov][square] for square in chess.SquareSet(my_pawns)])

    my_knights = board.knights & board.occupied_co[pov]
    my_knights_attack = chess.BB_EMPTY if not my_knights else \
        reduce(ior, [chess.BB_KNIGHT_ATTACKS[square] for square in chess.SquareSet(my_knights)])

    my_rooks = board.rooks & board.occupied_co[pov]
    my_rooks_attack = chess.BB_EMPTY if not my_rooks else \
        reduce(ior, [chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & board.occupied] |
                     chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(my_rooks)])

    my_bishops = board.bishops & board.occupied_co[pov]
    my_bishops_attack = chess.BB_EMPTY if not my_bishops else \
        reduce(ior, [chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(my_bishops)])

    my_queens = board.queens & board.occupied_co[pov]
    my_queens_attack = chess.BB_EMPTY if not my_queens else \
        reduce(ior, [chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & board.occupied] |
                     chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied] |
                     chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(my_queens)])

    my_king_attacks = chess.BB_KING_ATTACKS[board.king(pov)]

    supported_squares = (my_pawns_attack | my_knights_attack | my_rooks_attack |
                         my_bishops_attack | my_queens_attack | my_king_attacks)

    my_safe_pieces = board.occupied_co[pov] & ~op_pawns_attack & (supported_squares | safe_squares)

    # Count threatened quiet checks, considering only safe outposts that don't already check
    knight_checks = chess.BB_KNIGHT_ATTACKS[op_king]
    my_knights = board.knights & my_safe_pieces
    my_knights_attack = chess.BB_EMPTY if not my_knights else \
        reduce(ior, [chess.BB_KNIGHT_ATTACKS[square] for square in chess.SquareSet(my_knights)])
    knight_check_moves = count_set_bits(my_knights_attack & ~board.occupied & knight_checks)

    rook_checks = chess.BB_RANK_ATTACKS[op_king][rank_pieces] | chess.BB_FILE_ATTACKS[op_king][file_pieces]
    my_rooks = board.rooks & my_safe_pieces
    my_rooks_attack = chess.BB_EMPTY if not my_rooks else \
        reduce(ior, [chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & board.occupied] |
                     chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(my_rooks)])
    rook_check_moves = count_set_bits(my_rooks_attack & ~board.occupied & rook_checks)

    bishop_checks = chess.BB_DIAG_ATTACKS[op_king][diag_pieces]
    my_bishops = board.bishops & my_safe_pieces
    my_bishops_attack = chess.BB_EMPTY if not my_bishops else \
        reduce(ior, [chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(my_bishops)])
    bishop_check_moves = count_set_bits(my_bishops_attack & ~board.occupied & bishop_checks)

    queen_checks = (chess.BB_RANK_ATTACKS[op_king][rank_pieces] |
                    chess.BB_FILE_ATTACKS[op_king][file_pieces] |
                    chess.BB_DIAG_ATTACKS[op_king][diag_pieces])
    my_queens = board.queens & my_safe_pieces
    my_queens_attack = chess.BB_EMPTY if not my_queens else \
        reduce(ior, [chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & board.occupied] |
                     chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & board.occupied] |
                     chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & board.occupied]
                     for square in chess.SquareSet(my_queens)])
    queen_check_moves = count_set_bits(my_queens_attack & ~board.occupied & queen_checks)

    total_checks = knight_check_moves + rook_check_moves + bishop_check_moves + queen_check_moves

    # While all of these bitboards are available, look for attacks on hanging pieces
    op_hanging_pieces = board.occupied_co[not pov] & safe_squares
    reachable_hanging_pieces = op_hanging_pieces & supported_squares
    hanging_value = sum(
        count_set_bits(board.pieces_mask(piece, not pov) & reachable_hanging_pieces) * value
        for piece, value in PIECE_VALUES.items()
    )
    total_checks += hanging_value / 250

    my_hanging_material = sum(
        count_set_bits(board.pieces_mask(piece, pov) & ~supported_squares) * value
        for piece, value in PIECE_VALUES.items()
    )

    return total_checks, my_hanging_material
