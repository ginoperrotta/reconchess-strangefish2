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

import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from time import sleep, time
from typing import List

import chess.engine
import numpy as np
from reconchess import Square, Color
from tqdm import tqdm

from strangefish.strangefish_mht_core import StrangeFish
from strangefish.utilities import (
    SEARCH_SPOTS,
    stockfish,
    simulate_move,
    simulate_sense,
    generate_rbc_moves,
    generate_moves_without_opponent_pieces,
    force_promotion_to_queen,
    sense_partition_leq,
)
from strangefish.utilities.rbc_move_score import calculate_score, ScoreConfig

SCORE_ROUNDOFF = 1e-5
SENSE_SAMPLE_LIMIT = 1000
SCORE_SAMPLE_LIMIT = 300


@dataclass
class SenseConfig:
    boards_per_centipawn: float = 200  # The scaling factor for combining decision-impact and set-reduction sensing
    expected_outcome_coef: float = 1.0  # The scaling factor for sensing to maximize the expected turn outcome
    worst_outcome_coef: float = 0.0  # The scaling factor for sensing to maximize the worst turn outcome
    outcome_variance_coef: float = -0.5  # The scaling factor for sensing based on turn outcome variance
    score_variance_coef: float = 0.1  # The scaling factor for sensing based on move score variance


@dataclass
class MoveConfig:
    mean_score_factor: float = 0.7  # relative contribution of a move's average outcome on its compound score
    min_score_factor: float = 0.3  # relative contribution of a move's worst outcome on its compound score
    max_score_factor: float = 0.0  # relative contribution of a move's best outcome on its compound score
    threshold_score_factor: float = 0.0  # fraction below best compound score in which any move will be considered
    sense_by_move: bool = False  # Use bonus score to encourage board set reduction by attempted moves
    force_promotion_queen: bool = True  # change all pawn-promotion moves to choose queen, otherwise it's often a knight


@dataclass
class TimeConfig:
    turns_to_plan_for: int = 16  # fixed number of turns over which the remaining time will be divided
    min_time_for_turn: float = 1.0  # minimum time to allocate for a turn
    time_for_sense: float = 0.8  # fraction of turn spent in choose_sense
    time_for_move: float = 0.15  # fraction of turn spent in choose_move
    calc_time_per_move: float = 0.005  # starting time estimate for move score calculation


# Create a cache key for the requested board and move
def make_cache_key(board: chess.Board, move: chess.Move = None, prev_turn_score: int = None):
    return hash((board, move, prev_turn_score))


class StrangeFish2(StrangeFish):
    """
    The reconchess agent, StrangeFish2, that took second place in the NeurIPS 2021 tournament of RBC.

    StrangeFish2 is, clearly, an extension of the 2019 tournament winner, StrangeFish. Each tracks an exhaustive set of
    possible board states and chooses sense and move actions based on analysis of the scores assigned to each
    possibility. Scores are evaluated for pairs of (chess.Board, chess.Move) using the chess engine Stockfish plus a
    series of uncertainty-related heuristics.

    Moves are chosen to maximize the expected outcome of the current turn. This computation is configurable in
    MoveConfig; for the 2021 tournament, I used 70% average + 30% minimum score to evaluate move options.

    The primary difference between StrangeFish2 and its predecessor, StrangeFish, is the sense strategy. StrangeFish2
    chooses where to sense to maximize a hybrid objective that is primarily the same objective as the move strategy:
    the expected outcome of the current turn. To a lesser degree, the hybrid objective also includes the sense strategy
    of 2019 StrangeFish: maximize the expected change in move evaluation as a proxy for strategic information content.
    A third term is included: minimize the expected size of the resulting set of possible board states. This term is
    typically small, but if the set size grows large it can dominate the sense decision. In that case, StrangeFish2
    saves considerable computational effort by only computing this term, and not the first two.

    Additionally, StrangeFish2 has been significantly refactored from the original code (see that code for comparison
    at https://github.com/ginop/reconchess-strangefish). This framework is much more efficient, and bots previously
    based on StrangeFish would benefit from being ported over.

    To run StrangeFish2, you'll need to download the chess engine Stockfish from https://stockfishchess.org/download/
    and create an environment variable called STOCKFISH_EXECUTABLE that is the path to the downloaded Stockfish
    executable. Recent versions of Stockfish use a neural network for evaluation, which improves performance over the
    strictly rules-based prior versions, but seems to cause odd behavior when passed board states that are possible in
    RBC but not in chess. Or maybe it's unrelated to the network; either way, StrangeFish2 is only stable with versions
    up to Stockfish 11.
    """

    def __init__(
        self,
        log_to_file: bool = False,
        rc_disable_pbar: bool = False,
        sense_config: SenseConfig = SenseConfig(),
        move_config: MoveConfig = MoveConfig(),
        score_config: ScoreConfig = ScoreConfig(),
        time_config: TimeConfig = TimeConfig(),
        board_weight_90th_percentile: float = 5_000,
        min_board_weight: float = 0.01,
        while_we_wait_extension: bool = True,
    ):
        """
        Constructs an instance of the StrangeFish2 agent.

        :param log_to_file: A boolean flag to turn on/off logging to file gameLogs/<date code>.log
        :param rc_disable_pbar: A boolean flag to turn on/off the tqdm progress bars

        :param sense_config: A dataclass of parameters which determine the sense strategy's score calculation
        :param move_config: A dataclass of parameters which determine the move strategy's score calculation
        :param score_config: A dataclass of parameters which determine the score assigned to a board's strength
        :param time_config: A dataclass of parameters which determine how time is allocated between turns

        :param board_weight_90th_percentile: The centi-pawn score associated with a 0.9 weight in the board set
        :param min_board_weight: A lower limit on relative board weight `w = max(w, min_board_weight)`

        :param while_we_wait_extension: A bool that toggles the scoring of boards that could be reached two turns ahead
        """
        super().__init__(log_to_file, rc_disable_pbar)

        self.logger.debug("Creating new instance of StrangeFish2.")

        self.sense_config = sense_config
        self.move_config = move_config
        self.score_config = score_config
        self.time_config = time_config

        self.swap_sense_time = 90
        self.swap_sense_size = 20_000
        self.swap_sense_min_size = 150

        self.extra_move_time = False

        self.board_weight_90th_percentile = board_weight_90th_percentile
        self.min_board_weight = min_board_weight
        self.while_we_wait_extension = while_we_wait_extension

        # Initialize a list to store calculation time data for dynamic time management
        self.score_calc_times = []

        self.score_cache = dict()
        self.boards_in_cache = set()

        self.engine = None

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        super().handle_game_start(color, board, opponent_name)

        # At this point, the configuration can be adjusted based on the specific opponent.
        # For example:
        # if opponent_name == 'attacker':
        #     self.score_config.unsafe_attacker_penalty = 0
        #     self.score_config.search_check_in_three = True

        self.engine = stockfish.create_engine()

    def calc_time_per_move(self) -> float:
        """Estimate calculation time based on data stored so far this game (and a provided starting datum)"""
        n0 = 100
        t0 = self.time_config.calc_time_per_move * n0
        total_num = n0 + sum(n for n, t in self.score_calc_times)
        total_time = t0 + sum(t for n, t in self.score_calc_times)
        return total_time / total_num

    def allocate_time(self, seconds_left: float, fraction_turn_passed: float = 0):
        """Determine how much of the remaining time should be spent on (the rest of) the current turn."""
        turns_left = self.time_config.turns_to_plan_for - fraction_turn_passed  # account for previous parts of turn
        equal_time_split = seconds_left / turns_left
        return max(equal_time_split, self.time_config.min_time_for_turn)

    def weight_board_probability(self, score):
        """Convert a board strength score into a probability for use in weighted averages"""
        if self.board_weight_90th_percentile is None:
            return 1
        return 1 / (1 + np.exp(-2 * np.log(3) / self.board_weight_90th_percentile * score)) + self.min_board_weight

    def memo_calc_score(
        self,
        board: chess.Board,
        move: chess.Move = chess.Move.null(),
        prev_turn_score: int = None,
        key: int = None,
    ):
        """Memoized calculation of the score associated with one move on one board"""
        if key is None:
            key = make_cache_key(board, simulate_move(board, move), prev_turn_score)
        if key in self.score_cache:
            return self.score_cache[key], False

        score = calculate_score(
            board=board,
            move=move,
            prev_turn_score=prev_turn_score or 0,
            engine=self.engine,
            score_config=self.score_config,
            is_op_turn=prev_turn_score is None,
        )
        return score, True

    def memo_calc_set(self, requests):
        """Handler for requested scores. Filters for unique requests, then gets cached or calculated results."""

        filtered_requests = set()
        equivalent_requests = defaultdict(list)
        for board, move, prev_turn_score, pseudo_legal_moves in requests:
            if pseudo_legal_moves is None:
                pseudo_legal_moves = list(board.generate_pseudo_legal_moves())
            taken_move = simulate_move(board, move, pseudo_legal_moves)
            request_key = make_cache_key(board, move, prev_turn_score)
            result_key = make_cache_key(board, taken_move, prev_turn_score)
            equivalent_requests[result_key].append(request_key)
            filtered_requests.add((board, taken_move or chess.Move.null(), prev_turn_score, result_key))

        start = time()

        results = {}
        num_new = 0
        for board, move, prev_turn_score, key in filtered_requests:
            results[key], is_new = self.memo_calc_score(board, move, prev_turn_score, key=key)
            if is_new:
                num_new += 1

        for result_key, request_keys in equivalent_requests.items():
            for request_key in request_keys:
                result = {request_key: results[result_key]}
                self.score_cache.update(result)
                results.update(result)

        duration = time() - start
        if num_new:
            self.score_calc_times.append((num_new, duration))

        return results

    def cache_board(self, board: chess.Board):
        """Add a new board to the cache (evaluate the board's strength and relative score for every possible move)."""
        board.turn = not self.color
        op_score = self.memo_calc_set([(board, chess.Move.null(), None, None)])[make_cache_key(board)]
        board.turn = self.color
        pseudo_legal_moves = list(board.generate_pseudo_legal_moves())
        self.boards_in_cache.add(board)
        self.memo_calc_set(
            [(board, move, -op_score, pseudo_legal_moves) for move in generate_moves_without_opponent_pieces(board)]
        )

    def cache_favored_random_sample(self, sample_size):
        """Randomly sample from the board set, but also include all of the boards which are already in the cache."""
        pre_calculated_boards = self.boards & self.boards_in_cache
        priority_boards = (self.boards & self.priority_boards) - pre_calculated_boards

        sample_size_after_high_risk = sample_size - len(priority_boards)
        if sample_size_after_high_risk <= 0:
            return list(pre_calculated_boards) + random.sample(priority_boards, sample_size)
        else:
            remaining_boards = self.boards - pre_calculated_boards - priority_boards
            return (
                list(pre_calculated_boards)
                + list(priority_boards)
                + random.sample(
                    remaining_boards,
                    min(len(remaining_boards), sample_size_after_high_risk),
                )
            )

    def choose_uncached_board(self):
        """Randomly choose one board from next turn's board set, excluding boards which are already in the cache."""
        uncached_boards = (
            self.priority_boards - self.boards_in_cache or self.next_turn_boards_unsorted - self.boards_in_cache
        )
        return random.choice(tuple(uncached_boards)) if uncached_boards else None

    def sense_strategy(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        """Choose a square to sense. Delegate to one of two strategies based on current board set size."""
        if len(self.boards) <= self.swap_sense_min_size or (
            seconds_left > self.swap_sense_time and len(self.boards) < self.swap_sense_size
        ):
            return self.sense_max_outcome(sense_actions, moves, seconds_left)
        else:
            self.extra_move_time = True
            return self.sense_min_states(sense_actions, moves, seconds_left)

    def sense_min_states(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        """Choose a sense square to minimize the expected board set size."""

        # Don't sense if there is nothing to learn from it
        if len(self.boards) == 1:
            return None

        sample_size = min(len(self.boards), SENSE_SAMPLE_LIMIT)

        self.logger.debug(f"In sense phase with {seconds_left:.2f} seconds left. Set size is {sample_size}.")

        # Initialize some parameters for tracking information about possible sense results
        sense_results = defaultdict(Counter)

        # Get a random sampling of boards from the board set
        board_sample = random.sample(self.boards, sample_size)

        self.logger.debug(f"Sampled {len(board_sample)} boards out of {len(self.boards)} for sensing.")

        for board in tqdm(
            board_sample,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} " f"Minimizing expected set size from {len(self.boards)} boards",
            unit="boards",
        ):

            # Gather information about sense results for each square on each board
            for square in SEARCH_SPOTS:
                sense_result = simulate_sense(board, square)
                sense_results[square][sense_result] += 1

        # Calculate the expected board set reduction for each sense square (scale from board sample to full set)
        expected_set_reduction = {
            square:
                len(self.boards) *
                (1 - (1 / len(board_sample) ** 2) *
                 sum([sense_results[square][sense_result] ** 2 for sense_result in sense_results[square]]))
            for square in SEARCH_SPOTS
        }

        max_sense_score = max(expected_set_reduction.values())
        sense_choice = random.choice(
            [
                square
                for square, score in expected_set_reduction.items()
                if abs(score - max_sense_score) < SCORE_ROUNDOFF
            ]
        )
        return sense_choice

    def sense_max_outcome(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        """Choose a sense square to maximize the expected outcome of this turn."""

        # Don't sense if there is nothing to learn from it
        if len(self.boards) == 1:
            return None

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = self.allocate_time(seconds_left)
        time_for_phase = time_for_turn * self.time_config.time_for_sense
        time_per_move = self.calc_time_per_move()
        time_per_board = time_per_move * len(moves)
        sample_size = int(time_for_phase / time_per_board)

        self.logger.debug(
            "In sense phase with %.2f seconds left. Allocating %.2f seconds for this turn and "
            "%.2f seconds for this sense step. Estimating %.4f seconds per calc over %d moves is "
            "%.4f seconds per board score so we have time for %d boards.",
            seconds_left, time_for_turn, time_for_phase, time_per_move, len(moves), time_per_board, sample_size,
        )

        # Initialize some parameters for tracking information about possible sense results
        num_occurances = defaultdict(lambda: defaultdict(float))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        total_weighted_probability = 0
        sense_results = defaultdict(lambda: defaultdict(set))

        # Get a random sampling of boards from the board set
        board_sample = self.cache_favored_random_sample(sample_size)

        # Initialize arrays for board and move data (dictionaries work here, too, but arrays were faster)
        board_sample_weights = np.zeros(len(board_sample))
        move_scores = np.zeros([len(moves), len(board_sample)])

        self.logger.debug("Sampled %d boards out of %d for sensing.", len(board_sample), len(self.boards))

        # Get board position strengths before move for all boards in sample
        board_score_reqs = []
        for board in board_sample:
            board.turn = not self.color
            board_score_reqs.append((board, chess.Move.null(), None, None))
        board_score_dict = self.memo_calc_set(board_score_reqs)

        # Iterate over boards in sample, score all move options
        for num_board, board in enumerate(
            tqdm(
                board_sample,
                disable=self.rc_disable_pbar,
                desc=f"{chess.COLOR_NAMES[self.color]} "
                "Maximizing info for "
                f"{len(moves)} moves in {len(self.boards)} boards",
                unit="boards",
            )
        ):

            board.turn = not self.color
            op_score = board_score_dict[make_cache_key(board)]

            board_sample_weights[num_board] = self.weight_board_probability(op_score)
            total_weighted_probability += board_sample_weights[num_board]

            board.turn = self.color
            pseudo_legal_moves = list(board.generate_pseudo_legal_moves())

            move_score_dict = self.memo_calc_set(
                [(board, move, -op_score, pseudo_legal_moves) for move in moves]
            )  # Score all moves
            self.boards_in_cache.add(board)  # Record that this board (and all moves) are in our cache

            # Place move scores into array for later logical indexing
            for num_move, move in enumerate(moves):
                move_scores[num_move, num_board] = move_score_dict[make_cache_key(board, move, -op_score)]

            # Gather information about sense results for each square on each board (and king locations)
            for square in SEARCH_SPOTS:
                sense_result = simulate_sense(board, square)
                num_occurances[square][sense_result] += 1
                weighted_probability[square][sense_result] += board_sample_weights[num_board]
                sense_results[square][board] = sense_result

        # Calculate the mean, min, and max scores for each move across the board set (or at least the random sample)
        full_set_mean_scores = np.average(move_scores, axis=1, weights=board_sample_weights)
        full_set_min_scores = np.min(move_scores, axis=1)
        full_set_max_scores = np.max(move_scores, axis=1)
        # Combine the mean, min, and max changes in scores based on the config settings
        full_set_compound_score = (
            full_set_mean_scores * self.move_config.mean_score_factor
            + full_set_min_scores * self.move_config.min_score_factor
            + full_set_max_scores * self.move_config.max_score_factor
        )
        max_full_set_move_score = np.max(full_set_compound_score)
        full_set_move_choices = np.where(
            full_set_compound_score
            >= max_full_set_move_score - abs(max_full_set_move_score) * self.move_config.threshold_score_factor
        )[0]
        full_set_worst_outcome = np.min(move_scores[full_set_move_choices])

        possible_outcomes = move_scores[full_set_move_choices].flatten()
        outcome_weights = np.tile(board_sample_weights, len(full_set_move_choices)) / len(full_set_move_choices)
        full_set_expected_outcome = np.average(possible_outcomes, weights=outcome_weights)
        full_set_outcome_variance = np.sqrt(
            np.average(
                (possible_outcomes - full_set_expected_outcome) ** 2,
                weights=outcome_weights,
            )
        )

        sense_partitions = {
            square: {
                sense_result: {
                    i for i, board in enumerate(board_sample) if sense_result == sense_results[square][board]
                }
                for sense_result in set(sense_results[square].values())
            }
            for square in SEARCH_SPOTS
        }

        valid_sense_squares = set()
        for sense_option in SEARCH_SPOTS:
            # First remove elements of the current valid list if the new option is at least as good
            valid_sense_squares = [
                alt_option
                for alt_option in valid_sense_squares
                if not sense_partition_leq(
                    sense_partitions[sense_option].values(),
                    sense_partitions[alt_option].values(),
                )
            ]
            # Then add this new option if none of the current options dominate it
            if not any(
                sense_partition_leq(
                    sense_partitions[alt_option].values(),
                    sense_partitions[sense_option].values(),
                )
                for alt_option in valid_sense_squares
            ):
                valid_sense_squares.append(sense_option)

        # Find the expected change in move scores caused by any sense choice
        post_sense_score_variance = {}
        post_sense_move_uncertainty = {}
        post_sense_score_changes = {}
        post_sense_worst_outcome = {}
        post_sense_expected_outcome = {}
        post_sense_outcome_variance = {}
        for square in tqdm(
            valid_sense_squares,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} Evaluating sense impacts " f"for {len(self.boards)} boards",
            unit="squares",
        ):
            possible_results = set(sense_results[square].values())
            possible_outcomes = []
            outcome_weights = []
            if len(possible_results) > 1:
                post_sense_score_change = {}
                post_sense_move_choices = defaultdict(set)
                post_sense_move_scores = {}
                for sense_result in possible_results:
                    subset_index = list(sense_partitions[square][sense_result])
                    sub_set_move_scores = move_scores[:, subset_index]
                    sub_set_board_weights = board_sample_weights[subset_index]

                    # Calculate the mean, min, and max scores for each move across the board sub-set
                    sub_set_mean_scores = np.average(sub_set_move_scores, axis=1, weights=sub_set_board_weights)
                    sub_set_min_scores = np.min(sub_set_move_scores, axis=1)
                    sub_set_max_scores = np.max(sub_set_move_scores, axis=1)
                    sub_set_compound_score = (
                        sub_set_mean_scores * self.move_config.mean_score_factor
                        + sub_set_min_scores * self.move_config.min_score_factor
                        + sub_set_max_scores * self.move_config.max_score_factor
                    )
                    change_in_mean_scores = np.abs(sub_set_mean_scores - full_set_mean_scores)
                    change_in_min_scores = np.abs(sub_set_min_scores - full_set_min_scores)
                    change_in_max_scores = np.abs(sub_set_max_scores - full_set_max_scores)
                    compound_change_in_score = (
                        change_in_mean_scores * self.move_config.mean_score_factor
                        + change_in_min_scores * self.move_config.min_score_factor
                        + change_in_max_scores * self.move_config.max_score_factor
                    )

                    max_sub_set_move_score = np.max(sub_set_compound_score)
                    sub_set_move_choices = np.where(
                        sub_set_compound_score
                        >= max_sub_set_move_score
                        - abs(max_sub_set_move_score) * self.move_config.threshold_score_factor
                    )[0]

                    # Calculate the sense-choice-criteria
                    post_sense_move_scores[sense_result] = sub_set_compound_score
                    post_sense_score_change[sense_result] = float(np.mean(compound_change_in_score))
                    post_sense_move_choices[sense_result] = sub_set_move_choices

                    possible_outcomes.append(sub_set_move_scores[sub_set_move_choices].flatten())
                    outcome_weights.append(
                        np.tile(sub_set_board_weights, len(sub_set_move_choices)) / len(sub_set_move_choices)
                    )

                move_probabilities = {
                    move: sum(
                        weighted_probability[square][sense_result] / len(post_sense_move_choices[sense_result])
                        for sense_result in possible_results
                        if num_move in post_sense_move_choices[sense_result]
                    )
                    / sum(weighted_probability[square].values())
                    for num_move, move in enumerate(moves)
                }
                # assert np.isclose(sum(move_probabilities.values()), 1)

                _score_means = sum([
                    post_sense_move_scores[sense_result] * weighted_probability[square][sense_result]
                    for sense_result in set(sense_results[square].values())
                ]) / total_weighted_probability
                _score_variance = sum([
                    (post_sense_move_scores[sense_result] - _score_means) ** 2
                    * weighted_probability[square][sense_result]
                    for sense_result in set(sense_results[square].values())
                ]) / total_weighted_probability

                possible_outcomes = np.concatenate(possible_outcomes)
                outcome_weights = np.concatenate(outcome_weights)

                _outcome_means = np.average(possible_outcomes, weights=outcome_weights)
                _outcome_stdev = np.sqrt(
                    np.average(
                        (possible_outcomes - _outcome_means) ** 2,
                        weights=outcome_weights,
                    )
                )

                post_sense_score_variance[square] = np.average(np.sqrt(_score_variance))
                post_sense_move_uncertainty[square] = 1 - sum(p**2 for p in move_probabilities.values())
                post_sense_score_changes[square] = sum([
                    post_sense_score_change[sense_result] * weighted_probability[square][sense_result]
                    for sense_result in set(sense_results[square].values())
                ]) / total_weighted_probability
                post_sense_worst_outcome[square] = min(possible_outcomes)
                post_sense_expected_outcome[square] = _outcome_means
                post_sense_outcome_variance[square] = _outcome_stdev

            else:
                post_sense_score_variance[square] = 0
                post_sense_move_uncertainty[square] = 0
                post_sense_score_changes[square] = 0
                post_sense_worst_outcome[square] = full_set_worst_outcome
                post_sense_expected_outcome[square] = full_set_expected_outcome
                post_sense_outcome_variance[square] = full_set_outcome_variance

        # Also calculate the expected board set reduction for each sense square (scale from board sample to full set)
        expected_set_reduction = {
            square:
                len(self.boards) *
                (1 - (1 / len(board_sample) / total_weighted_probability) *
                 sum([num_occurances[square][sense_result] * weighted_probability[square][sense_result]
                      for sense_result in set(sense_results[square].values())]))
            for square in SEARCH_SPOTS
        }

        # Combine the expected-outcome, score-variance, and set-reduction estimates
        sense_score = {
            square: post_sense_expected_outcome[square] * self.sense_config.expected_outcome_coef
            + post_sense_worst_outcome[square] * self.sense_config.worst_outcome_coef
            + post_sense_outcome_variance[square] * self.sense_config.outcome_variance_coef
            + post_sense_score_variance[square] * self.sense_config.score_variance_coef
            + (expected_set_reduction[square] / self.sense_config.boards_per_centipawn)
            for square in valid_sense_squares
        }

        self.logger.debug(
            "Sense values presented as: (score change, move uncertainty, score variance, "
            "expected outcome, outcome variance, worst outcome, set reduction, "
            "and final combined score)"
        )
        for square in valid_sense_squares:
            self.logger.debug(
                f"{chess.SQUARE_NAMES[square]}: ("
                f"{post_sense_score_changes[square]:5.0f}, "
                f"{post_sense_move_uncertainty[square]:.2f}, "
                f"{post_sense_score_variance[square]:5.0f}, "
                f"{post_sense_expected_outcome[square]: 5.0f}, "
                f"{post_sense_outcome_variance[square]:5.0f}, "
                f"{post_sense_worst_outcome[square]: 5.0f}, "
                f"{expected_set_reduction[square]:5.0f}, "
                f"{sense_score[square]: 5.0f})"
            )

        # Determine the minimum score a move needs to be considered
        highest_score = max(sense_score.values())
        threshold_score = highest_score - SCORE_ROUNDOFF
        # Create a list of all moves which scored above the threshold
        sense_options = [square for square, score in sense_score.items() if score >= threshold_score]
        # Randomly choose one of the remaining options
        sense_choice = random.choice(sense_options)
        return sense_choice

    def move_strategy(self, moves: List[chess.Move], seconds_left: float):
        """
        Choose the move with the maximum score calculated from a combination of mean, min, and max possibilities.

        This strategy randomly samples from the current board set, then weights the likelihood of each board being the
        true state by an estimate of the opponent's position's strength. Each move is scored on each board, and the
        resulting scores are assessed together by looking at the worst-case score, the average score, and the best-case
        score. The relative contributions of these components to the compound score are determined by a config object.
        If requested by the config, bonus points are awarded to moves based on the expected number of boards removed
        from the possible set by attempting that move. Deterministic move patterns are reduced by randomly choosing a
        move that is within a few percent of the maximum score.
        """

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = self.allocate_time(seconds_left, fraction_turn_passed=1 - self.time_config.time_for_move)
        if self.extra_move_time:
            time_for_phase = time_for_turn * self.time_config.time_for_sense
            self.extra_move_time = False
        else:
            time_for_phase = time_for_turn * self.time_config.time_for_move
        time_per_move = self.calc_time_per_move()
        time_per_board = time_per_move * len(moves)
        sample_size = max(1, int(time_for_phase / time_per_board))

        self.logger.debug(
            "In move phase with %.2f seconds left. Allowing up to %.2f seconds for this move step. Estimating %.4f "
            "seconds per calc over %d moves is %.4f seconds per board score so we have time for %d boards.",
            seconds_left,
            time_for_turn,
            time_per_move,
            len(moves),
            time_per_board,
            sample_size,
        )

        move_scores = defaultdict(list)
        weighted_sum_move_scores = defaultdict(float)

        # Initialize some parameters for tracking information about possible move results
        num_occurances = defaultdict(lambda: defaultdict(int))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        move_possibilities = defaultdict(set)
        total_weighted_probability = 0

        # Get a random sampling of boards from the board set
        board_sample = self.cache_favored_random_sample(sample_size)
        self.logger.debug(
            "Sampled %d boards out of %d for moving.",
            len(board_sample),
            len(self.boards),
        )

        # Get board position strengths before move for all boards in sample
        board_score_reqs = []
        for board in board_sample:
            board.turn = not self.color
            board_score_reqs.append((board, chess.Move.null(), None, None))

        board_score_dict = self.memo_calc_set(board_score_reqs)

        for board in tqdm(
            board_sample,
            disable=self.rc_disable_pbar,
            desc=f"{chess.COLOR_NAMES[self.color]} Calculating choose_move scores "
            f"{len(moves)} moves in {len(self.boards)} boards",
            unit="boards",
        ):

            board.turn = not self.color
            op_score = board_score_dict[make_cache_key(board)]
            board_weight = self.weight_board_probability(op_score)
            total_weighted_probability += board_weight

            board.turn = self.color
            pseudo_legal_moves = list(board.generate_pseudo_legal_moves())

            move_score_dict = self.memo_calc_set(
                [(board, move, -op_score, pseudo_legal_moves) for move in moves]
            )  # Score all moves
            self.boards_in_cache.add(board)  # Record that this board (and all moves) are in our cache

            # Gather scores and information about move results for each requested move on each board
            for move in moves:
                score = move_score_dict[make_cache_key(board, move, -op_score)]

                sim_move = simulate_move(board, move, pseudo_legal_moves) or chess.Move.null()
                is_capture = board.is_capture(sim_move)

                move_scores[move].append(score)
                weighted_sum_move_scores[move] += score * board_weight

                move_result = (sim_move, is_capture)
                move_possibilities[move].add(move_result)
                num_occurances[move][move_result] += 1
                weighted_probability[move][move_result] += board_weight

        # Combine the mean, min, and max possible scores based on config settings
        compound_score = {
            move: (
                weighted_sum_move_scores[move] / total_weighted_probability * self.move_config.mean_score_factor
                + min(scores) * self.move_config.min_score_factor
                + max(scores) * self.move_config.max_score_factor
            )
            for (move, scores) in move_scores.items()
        }

        # Add centipawn points to a move based on an estimate of the board set reduction caused by that move
        if self.move_config.sense_by_move:
            compound_score = {
                move: score
                + 1
                / self.sense_config.boards_per_centipawn
                * len(self.boards)
                * (
                    1
                    - (1 / len(board_sample) / total_weighted_probability)
                    * sum(
                        [
                            num_occurances[move][move_result] * weighted_probability[move][move_result]
                            for move_result in move_possibilities[move]
                        ]
                    )
                )
                for move, score in compound_score.items()
            }

        # Determine the minimum score a move needs to be considered
        highest_score = max(compound_score.values())
        threshold_score = highest_score - abs(highest_score) * self.move_config.threshold_score_factor

        # Create a list of all moves which scored above the threshold
        move_options = [move for move, score in compound_score.items() if score >= threshold_score]
        # Eliminate move options which we know to be illegal (mainly for replay clarity)
        move_options = [
            move for move in move_options if move in {taken_move for taken_move, _ in move_possibilities[move]}
        ]
        # Randomly choose one of the remaining moves
        move_choice = random.choice(move_options)

        return force_promotion_to_queen(move_choice) if self.move_config.force_promotion_queen else move_choice

    def downtime_strategy(self):
        """
        Calculate scores for moves on next turn's boards. Store to cache for later processing acceleration.
        """
        uncached_board = self.choose_uncached_board()

        # If there are still boards for next turn without scores calculated, calculate move scores for one
        if uncached_board:
            if uncached_board.king(chess.WHITE) is not None and uncached_board.king(chess.BLACK) is not None:
                self.cache_board(uncached_board)
            else:
                self.logger.debug(f"Requested board scores when king was missing! {uncached_board}")

        # Otherwise, calculate move scores for a random board that could be reached in two turns
        elif self.while_we_wait_extension:
            board = random.choice(tuple(self.next_turn_boards_unsorted))
            board.push(random.choice(list(generate_rbc_moves(board))))
            board.push(random.choice(list(generate_rbc_moves(board))))
            if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
                self.cache_board(board)

        else:
            sleep(0.001)

    def gameover_strategy(self):
        """
        Quit the StockFish engine instance(s) associated with this strategy once the game is over.
        """
        self.logger.debug(
            "During this game, averaged %.5f seconds per score using search depth %d.",
            self.calc_time_per_move(),
            self.score_config.search_depth,
        )

        # Shut down StockFish
        self.logger.debug("Terminating engine.")
        self.engine.quit()
        self.logger.debug("Engine exited.")
