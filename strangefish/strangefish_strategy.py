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
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from time import sleep, time
from typing import List, Tuple, Optional

import chess.engine
import numpy as np
from reconchess import Square, Color
from tqdm import tqdm

from strangefish.strangefish_mht_core import StrangeFish, RC_DISABLE_PBAR
from strangefish.utilities import (
    SEARCH_SPOTS,
    stockfish,
    simulate_move,
    sense_masked_bitboards,
    rbc_legal_moves,
    rbc_legal_move_requests,
    sense_partition_leq,
    PASS,
    fast_copy_board,
    print_sense_result,
)
from strangefish.utilities.player_logging import create_file_handler
from strangefish.utilities.rbc_move_score import calculate_score, ScoreConfig, ENGINE_CACHE_STATS


SCORE_ROUNDOFF = 1e-5
SENSE_SAMPLE_LIMIT = 2500
SCORE_SAMPLE_LIMIT = 250


@dataclass
class RunningEst:
    num_samples: int = 0
    total_weight: float = 0
    minimum: float = None
    maximum: float = None
    average: float = None

    def update(self, value, weight=1):
        self.num_samples += 1
        self.total_weight += weight
        if self.num_samples == 1:
            self.minimum = self.maximum = self.average = value
        else:
            self.average += (value - self.average) * weight / self.total_weight
            if value < self.minimum:
                self.minimum = value
            elif value > self.maximum:
                self.maximum = value


@dataclass
class SenseConfig:
    boards_per_centipawn: float = 50  # The scaling factor for combining decision-impact and set-reduction sensing
    expected_outcome_coef: float = 1.0  # The scaling factor for sensing to maximize the expected turn outcome
    worst_outcome_coef: float = 0.2  # The scaling factor for sensing to maximize the worst turn outcome
    outcome_variance_coef: float = -0.3  # The scaling factor for sensing based on turn outcome variance
    score_variance_coef: float = 0.15  # The scaling factor for sensing based on move score variance


@dataclass
class MoveConfig:
    mean_score_factor: float = 0.7  # relative contribution of a move's average outcome on its compound score
    min_score_factor: float = 0.3  # relative contribution of a move's worst outcome on its compound score
    max_score_factor: float = 0.0  # relative contribution of a move's best outcome on its compound score
    threshold_score: float = 10  # centipawns below best compound score in which any move will be considered
    sense_by_move: bool = False  # Use bonus score to encourage board set reduction by attempted moves
    force_promotion_queen: bool = True  # change all pawn-promotion moves to choose queen, otherwise it's often a knight
    sampling_exploration_coef: float = 1_000.0  # UCT coef for sampling moves to eval  # TODO: tune
    move_sample_rep_limit: int = 100  # Number of consecutive iterations with same best move before breaking loop  # TODO: tune


@dataclass
class TimeConfig:
    turns_to_plan_for: int = 16  # fixed number of turns over which the remaining time will be divided
    min_time_for_turn: float = 3.0  # minimum time to allocate for a turn
    max_time_for_turn: float = 40.0  # maximum time to allocate for a turn
    time_for_sense: float = 0.7  # fraction of turn spent in choose_sense
    time_for_move: float = 0.3  # fraction of turn spent in choose_move
    calc_time_per_move: float = 0.005  # starting time estimate for move score calculation


# Create a cache key for the requested board and move
def make_cache_key(board: chess.Board, move: chess.Move = PASS, prev_turn_score: int = None):
    return board, move, prev_turn_score


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

        log_to_file=True,
        game_id=None,
        rc_disable_pbar=RC_DISABLE_PBAR,

        load_score_cache: bool = True,
        load_opening_book: bool = True,

        sense_config: SenseConfig = SenseConfig(),
        move_config: MoveConfig = MoveConfig(),
        score_config: ScoreConfig = ScoreConfig(),
        time_config: TimeConfig = TimeConfig(),

        board_weight_90th_percentile: float = 3_000,
        min_board_weight: float = 0.02,

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
        super().__init__(log_to_file=log_to_file, game_id=game_id, rc_disable_pbar=rc_disable_pbar)

        self.logger.debug("Creating new instance of StrangeFish2.")

        engine_logger = logging.getLogger("chess.engine")
        engine_logger.setLevel(logging.DEBUG)
        file_handler = create_file_handler(f"engine_logs/game_{game_id}_engine.log", 10_000)
        engine_logger.addHandler(file_handler)
        engine_logger.debug("File handler added to chess.engine logs")

        self.load_score_cache = load_score_cache
        self.load_opening_book = load_opening_book
        self.opening_book = None

        self.sense_config = sense_config
        self.move_config = move_config
        self.score_config = score_config
        self.time_config = time_config

        self.swap_sense_time = 90
        self.swap_sense_size = 10_000
        self.swap_sense_min_size = 150

        self.time_switch_aggro = 60

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

        if self.load_score_cache:
            self.logger.debug("Loading cached scores from file")
            try:
                with open(f"{'w' if color is chess.WHITE else 'b'}_cache.pk", "rb") as file:
                    self.boards_in_cache, self.score_cache = pickle.load(file)
            except FileNotFoundError:
                self.logger.warning(
                    "Parameter `load_score_cache` is True, "
                    f"but the score cache file, {'w' if color is chess.WHITE else 'b'}_cache.pk, was not found! "
                    "Playing without it."
                )
            else:
                self.logger.debug(
                    f"Loaded {len(self.score_cache)} cached scores for {len(self.boards_in_cache)} positions"
                )

        if self.load_opening_book:
            self.logger.debug("Loading opening book from file")
            try:
                opening_book_filename = (
                    f"opening_books/opening"
                    f"_{'W' if color is chess.WHITE else 'B'}"
                    f"_{opponent_name}.pk"
                )
                with open(opening_book_filename, "rb") as file:
                    self.opening_book = pickle.load(file)
            except FileNotFoundError:
                self.logger.debug(f"No opening book for {opponent_name}, loading default book")
                opening_book_filename = (
                    f"opening_books/opening"
                    f"_{'W' if color is chess.WHITE else 'B'}"
                    f"_ALL.pk"
                )
                try:
                    with open(opening_book_filename, "rb") as file:
                        self.opening_book = pickle.load(file)
                except FileNotFoundError:
                    self.logger.warning(
                        "Parameter `load_opening_book` is True, "
                        f"but the default opening book, {opening_book_filename}, was not found! "
                        "Playing without it."
                    )
                else:
                    self.logger.debug(f"Loaded opening book: {opening_book_filename}")
            else:
                self.logger.debug(f"Loaded opening book: {opening_book_filename}")

        # Set a few modifications for specific opponents:
        if opponent_name in ('random', "RandomBot", 'attacker', "AttackerBot"):
            self.board_weight_90th_percentile = None
            self.score_config.require_sneak = False
            self.score_config.capture_king_score = 20_000
            self.score_config.into_check_score = -10_000
            self.score_config.reward_attacker = 2_000
            self.score_config.passing_reward = -100
            self.score_config.capture_penalty = 100
            self.score_config.search_depth = 4
        if opponent_name in ('trout', "TroutBot"):
            self.score_config.search_depth = 6
            self.score_config.reward_attacker = 30
            self.score_config.capture_penalty = 10
            self.score_config.passing_reward = -10
            self.min_board_weight = 0.1
            self.score_config.reward_check_threats = 30
            self.score_config.hanging_material_penalty_ratio = 0.2
            self.score_config.safe_pawn_pressure_reward = 25

        self.engine = stockfish.create_engine()

    def calc_time_per_move(self) -> float:
        """Estimate calculation time based on data stored so far this game (and a provided starting datum)"""
        n0 = 1
        t0 = self.time_config.calc_time_per_move * n0
        total_num = n0 + sum(n for n, t in self.score_calc_times)
        total_time = t0 + sum(t for n, t in self.score_calc_times)
        return total_time / total_num

    def allocate_time(self, seconds_left: float, fraction_turn_passed: float = 0):
        """Determine how much of the remaining time should be spent on (the rest of) the current turn."""
        turns_left = self.time_config.turns_to_plan_for - fraction_turn_passed  # account for previous parts of turn
        equal_time_split = seconds_left / turns_left
        return min(max(equal_time_split, self.time_config.min_time_for_turn), self.time_config.max_time_for_turn)

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
        key = None,
    ):
        """Memoized calculation of the score associated with one move on one board"""
        if key is None:
            key = make_cache_key(board, simulate_move(board, move) or PASS, prev_turn_score)
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
            taken_move = simulate_move(board, move, pseudo_legal_moves) or PASS
            request_key = make_cache_key(board, move, prev_turn_score)
            result_key = make_cache_key(board, taken_move, prev_turn_score)
            equivalent_requests[result_key].append(request_key)
            filtered_requests.add((board, taken_move, prev_turn_score, result_key))

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
        op_score = self.memo_calc_set([(board, PASS, None, None)])[make_cache_key(board)]
        pseudo_legal_moves = list(board.generate_pseudo_legal_moves())
        self.boards_in_cache.add(board)
        self.memo_calc_set([(board, move, -op_score, pseudo_legal_moves) for move in rbc_legal_move_requests(board)])

    def choose_uncached_board(self):
        """Randomly choose one board from next turn's board set, excluding boards which are already in the cache."""
        # Sample from prioritized list in descending order
        for priority in sorted(self.board_sample_priority.keys(), reverse=True):
            uncached_boards = self.board_sample_priority[priority] - self.boards_in_cache
            if uncached_boards:
                return random.choice(tuple(uncached_boards))
        return None

    def last_ditch_plan(self):
        self.logger.info('Time is running out! Switching to aggressive backup plan.')

        self.sense_config.expected_outcome_coef = 1.0
        self.sense_config.worst_outcome_coef = 0.0
        self.sense_config.outcome_variance_coef *= 0.1
        self.sense_config.score_variance_coef *= 0.1

        self.move_config.max_score_factor = 0.0
        self.move_config.mean_score_factor = 1.0
        self.move_config.min_score_factor = 0
        self.move_config.threshold_score *= 10
        self.move_config.sense_by_move = False

        self.swap_sense_size = 1_000

        self.score_config.require_sneak = False
        self.score_config.reward_attacker *= 10
        self.score_config.unsafe_attacker_penalty *= 3
        self.score_config.capture_king_score *= 10
        self.score_config.into_check_score /= 2
        self.score_config.capture_penalty = max(20., self.score_config.capture_penalty)
        self.score_config.passing_reward = min(-20., self.score_config.passing_reward)

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        super().handle_opponent_move_result(captured_my_piece, capture_square)
        if self.opening_book is not None:
            capture_square_str = "-" if capture_square is None else chess.SQUARE_NAMES[capture_square]
            if capture_square in self.opening_book:
                self.logger.debug(f"Op cap square {capture_square_str} is in opening book")
                self.opening_book = self.opening_book[capture_square]
            else:
                self.logger.debug(f"Op cap square {capture_square_str} is not in opening book")
                self.opening_book = None

    def sense_strategy(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        """Choose a square to sense. Delegate to one of two strategies based on current board set size."""

        if self.opening_book is not None:
            # Make initial choices out of opening book
            options = tuple(self.opening_book)
            sense_choice = random.choice(options)
            self.logger.debug(
                f"Selected opening book sense {'-' if sense_choice is None else chess.SQUARE_NAMES[sense_choice]}"
                f" from options [{' '.join('-' if sq is None else chess.SQUARE_NAMES[sq] for sq in options)}]"
            )
            self.opening_book = self.opening_book[sense_choice]
            if self.opening_book is None:
                self.logger.debug("Reached the end of this opening book sequence")
            return sense_choice

        if seconds_left < self.time_switch_aggro:
            self.time_switch_aggro = -100
            self.last_ditch_plan()

        # Don't sense if there is nothing to learn from it
        if len(self.boards) == 1:
            return None

        n_unscored_boards = len(self.boards - self.boards_in_cache)
        if n_unscored_boards <= self.swap_sense_min_size or (seconds_left > self.swap_sense_time and n_unscored_boards < self.swap_sense_size):
            return self.sense_max_outcome(sense_actions, moves, seconds_left)
        else:
            self.extra_move_time = True
            return self.sense_min_states(sense_actions, moves, seconds_left)

    def sense_min_states(self, sense_actions: List[Square], moves: List[chess.Move], seconds_left: float):
        """Choose a sense square to minimize the expected board set size."""

        sample_size = min(len(self.boards), SENSE_SAMPLE_LIMIT)

        self.logger.debug(f"In sense phase with {seconds_left:.2f} seconds left. Set size is {sample_size}.")

        # Initialize some parameters for tracking information about possible sense results
        num_occurances = defaultdict(lambda: defaultdict(float))
        sense_results = defaultdict(lambda: defaultdict(set))
        sense_possibilities = defaultdict(set)

        # Get a random sampling of boards from the board set
        board_sample = random.sample(self.boards, sample_size)

        self.logger.debug(f"Sampled {len(board_sample)} boards out of {len(self.boards)} for sensing.")

        for board in tqdm(board_sample, disable=self.rc_disable_pbar,
                          desc="Sense quantity evaluation", unit="boards"):

            # Gather information about sense results for each square on each board
            for square in SEARCH_SPOTS:
                sense_result = sense_masked_bitboards(board, square)
                num_occurances[square][sense_result] += 1
                sense_results[board][square] = sense_result
                sense_possibilities[square].add(sense_result)

        # Calculate the expected board set reduction for each sense square (scale from board sample to full set)
        expected_set_reduction = {
            square:
                len(self.boards) *
                (1 - (1 / len(board_sample) ** 2) *
                 sum([num_occurances[square][sense_result] ** 2 for sense_result in sense_possibilities[square]]))
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

        if self.move_config.force_promotion_queen:
            moves = [move for move in moves if move.promotion in (None, chess.QUEEN)]

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = self.allocate_time(seconds_left)
        time_for_phase = time_for_turn * self.time_config.time_for_sense

        self.logger.debug(f"In sense phase with {seconds_left:.2f} seconds left. "
                          f"Allowing up to {time_for_phase:.2f} seconds for this sense step.")

        # Until stop time, compute board scores
        n_boards = len(self.boards)
        start_time = time()
        phase_end_time = start_time + time_for_phase
        board_sample = self.boards & self.boards_in_cache
        num_precomputed = len(board_sample)
        with tqdm(desc="Computing move scores", unit="boards", disable=self.rc_disable_pbar, total=len(self.boards)) as pbar:
            pbar.update(num_precomputed)
            for priority in sorted(self.board_sample_priority.keys(), reverse=True):
                priority_boards = self.boards & self.board_sample_priority[priority]
                if not priority_boards:
                    continue
                num_priority = len(priority_boards)
                self.logger.debug(
                    f"Sampling from priority {priority}: {num_priority} boards ({num_priority/n_boards*100:0.0f}%)"
                )
                priority_boards -= self.boards_in_cache
                while priority_boards and len(board_sample) < SCORE_SAMPLE_LIMIT and time() < phase_end_time:
                    board = priority_boards.pop()
                    self.cache_board(board)
                    board_sample.add(board)
                    pbar.update()
        self.logger.debug(f"Spent {time() - start_time:0.1f} seconds computing new move scores")
        self.logger.debug(f"Had {num_precomputed} of {len(self.boards)} boards already cached")
        self.logger.debug(f"Sampled {len(board_sample)} of {len(self.boards)} boards for sensing.")

        # Initialize some parameters for tracking information about possible sense results
        num_occurances = defaultdict(lambda: defaultdict(float))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        total_weighted_probability = 0
        sense_results = defaultdict(lambda: defaultdict(set))
        sense_partitions = {sq: defaultdict(set) for sq in SEARCH_SPOTS}

        # Initialize arrays for board and move data (dictionaries work here, too, but arrays were faster)
        board_sample_weights = np.zeros(len(board_sample))
        move_scores = np.zeros([len(moves), len(board_sample)])

        for num_board, board in enumerate(tqdm(board_sample, disable=self.rc_disable_pbar,
                                               desc="Sense+Move outcome evaluation", unit="boards")):

            op_score = self.score_cache[make_cache_key(board)]

            board_sample_weights[num_board] = self.weight_board_probability(op_score)
            total_weighted_probability += board_sample_weights[num_board]

            # Place move scores into array for later logical indexing
            for num_move, move in enumerate(moves):
                move_scores[num_move, num_board] = self.score_cache[make_cache_key(board, move, -op_score)]

            # Gather information about sense results for each square on each board (and king locations)
            for square in SEARCH_SPOTS:
                sense_result = sense_masked_bitboards(board, square)
                num_occurances[square][sense_result] += 1
                weighted_probability[square][sense_result] += board_sample_weights[num_board]
                sense_results[square][board] = sense_result
                sense_partitions[square][sense_result].add(num_board)

        # Calculate the mean, min, and max scores for each move across the board set (or at least the random sample)
        full_set_mean_scores = (np.average(move_scores, axis=1, weights=board_sample_weights))
        full_set_min_scores = (np.min(move_scores, axis=1))
        full_set_max_scores = (np.max(move_scores, axis=1))
        # Combine the mean, min, and max changes in scores based on the config settings
        full_set_compound_score = (
                full_set_mean_scores * self.move_config.mean_score_factor +
                full_set_min_scores * self.move_config.min_score_factor +
                full_set_max_scores * self.move_config.max_score_factor
        )
        max_full_set_move_score = np.max(full_set_compound_score)
        full_set_move_choices = np.where(
            full_set_compound_score >= (max_full_set_move_score - self.move_config.threshold_score)
        )[0]
        full_set_worst_outcome = np.min(move_scores[full_set_move_choices])

        possible_outcomes = move_scores[full_set_move_choices].flatten()
        outcome_weights = np.tile(board_sample_weights, len(full_set_move_choices)) / len(full_set_move_choices)
        full_set_expected_outcome = np.average(possible_outcomes, weights=outcome_weights)
        full_set_outcome_variance = np.sqrt(np.average((possible_outcomes - full_set_expected_outcome) ** 2, weights=outcome_weights))

        valid_sense_squares = set()
        for sense_option in SEARCH_SPOTS:
            # First remove elements of the current valid list if the new option is at least as good
            valid_sense_squares = [alt_option for alt_option in valid_sense_squares
                                   if not sense_partition_leq(sense_partitions[sense_option].values(),
                                                              sense_partitions[alt_option].values())]
            # Then add this new option if none of the current options dominate it
            if not any(
                sense_partition_leq(sense_partitions[alt_option].values(), sense_partitions[sense_option].values())
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
        for square in tqdm(valid_sense_squares, disable=self.rc_disable_pbar,
                           desc="Evaluating sense options", unit="squares"):
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
                    sub_set_mean_scores = (np.average(sub_set_move_scores, axis=1, weights=sub_set_board_weights))
                    sub_set_min_scores = (np.min(sub_set_move_scores, axis=1))
                    sub_set_max_scores = (np.max(sub_set_move_scores, axis=1))
                    sub_set_compound_score = (
                            sub_set_mean_scores * self.move_config.mean_score_factor +
                            sub_set_min_scores * self.move_config.min_score_factor +
                            sub_set_max_scores * self.move_config.max_score_factor
                    )
                    change_in_mean_scores = np.abs(sub_set_mean_scores - full_set_mean_scores)
                    change_in_min_scores = np.abs(sub_set_min_scores - full_set_min_scores)
                    change_in_max_scores = np.abs(sub_set_max_scores - full_set_max_scores)
                    compound_change_in_score = (
                            change_in_mean_scores * self.move_config.mean_score_factor +
                            change_in_min_scores * self.move_config.min_score_factor +
                            change_in_max_scores * self.move_config.max_score_factor
                    )

                    max_sub_set_move_score = np.max(sub_set_compound_score)
                    sub_set_move_choices = np.where(
                        sub_set_compound_score >= (max_sub_set_move_score - self.move_config.threshold_score)
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
                post_sense_move_uncertainty[square] = 1 - sum(p ** 2 for p in move_probabilities.values())
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
                (len(self.boards) + self.stored_old_boards.expected_size) *
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

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        super().handle_sense_result(sense_result)
        if self.opening_book is not None:
            sense_result = tuple((sq, p) for sq, p in sense_result)
            sense_result_str = print_sense_result(sense_result)
            if sense_result in self.opening_book:
                self.logger.debug(f"Sense result {sense_result_str} is in opening book")
                self.opening_book = self.opening_book[sense_result]
            else:
                self.logger.debug(f"Sense result {sense_result_str} is not in opening book")
                self.opening_book = None

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

        if self.opening_book is not None:
            # Make initial choices out of opening book
            options = tuple(self.opening_book)
            move_choice = random.choice(options)
            self.logger.debug(f"Selected opening book move {move_choice.uci()}"
                              f" from options [{' '.join(move.uci() for move in options)}]")
            self.opening_book = self.opening_book[move_choice]
            if self.opening_book is None:
                self.logger.debug("Reached the end of this opening book sequence")
            return move_choice

        if len(self.boards) == 1 and random.random() < 0.9:  # 10% chance to still use normal move method
            return self._get_engine_move(next(iter(self.boards)))

        if seconds_left < self.time_switch_aggro:
            self.time_switch_aggro = -100
            self.last_ditch_plan()

        if self.move_config.force_promotion_queen:
            moves = [move for move in moves if move.promotion in (None, chess.QUEEN)]

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = self.allocate_time(seconds_left)
        if self.extra_move_time:
            time_for_phase = time_for_turn
            self.extra_move_time = False
        else:
            time_for_phase = time_for_turn * self.time_config.time_for_move

        self.logger.debug(f"In move phase with {seconds_left:.2f} seconds left. "
                          f"Allowing up to {time_for_phase:.2f} seconds for this move step.")

        # First compute valid move requests and taken move -> requested move mappings
        possible_move_requests = set(moves)
        valid_move_requests = set()
        all_maps_to_taken_move = {}
        all_maps_from_taken_move = {}
        for board in tqdm(self.boards, desc="Writing move maps", unit="boards", disable=self.rc_disable_pbar):
            legal_moves = set(rbc_legal_moves(board))
            valid_move_requests |= legal_moves
            map_to_taken_move = {}
            map_from_taken_move = defaultdict(set)
            for requested_move in moves:
                taken_move = simulate_move(board, requested_move, legal_moves) or PASS
                map_to_taken_move[requested_move] = taken_move
                map_from_taken_move[taken_move].add(requested_move)
            all_maps_to_taken_move[board] = map_to_taken_move
            all_maps_from_taken_move[board] = map_from_taken_move
        # Filter main list of moves and all move maps
        moves = possible_move_requests & valid_move_requests
        all_maps_from_taken_move = {
            board:
                {
                    taken_move: requested_moves & moves
                    for taken_move, requested_moves in map_from_taken_move.items()
                }
            for board, map_from_taken_move in all_maps_from_taken_move.items()
        }

        # Initialize move score estimates and populate with any pre-computed scores
        move_scores = defaultdict(RunningEst)
        boards_to_sample = {move: set() for move in moves}
        for board in tqdm(self.boards, desc="Reading pre-computed move scores", unit="boards", disable=self.rc_disable_pbar):
            try:
                score_before_move = self.score_cache[make_cache_key(board)]
            except KeyError:
                for move in moves:
                    boards_to_sample[move].add(board)
            else:
                weight = self.weight_board_probability(score_before_move)
                for taken_move, requested_moves in all_maps_from_taken_move[board].items():
                    try:
                        score = self.score_cache[make_cache_key(board, taken_move, -score_before_move)]
                    except KeyError:
                        for move in requested_moves:
                            boards_to_sample[move].add(board)
                    else:
                        for move in requested_moves:
                            move_scores[move].update(score, weight)
        incomplete_moves = {move for move in moves if boards_to_sample[move]}

        # Until stop time, compute board scores
        start_time = time()
        phase_end_time = start_time + time_for_phase
        total_evals = len(self.boards) * len(moves)  # TODO: change this to match new mapping
        num_evals_done = num_precomputed = sum(est.num_samples for est in move_scores.values())
        with tqdm(desc="Computing move scores", unit="evals", disable=self.rc_disable_pbar, total=total_evals) as pbar:
            pbar.update(num_precomputed)

            sorted_priorities = sorted(self.board_sample_priority.keys(), reverse=True)
            top_move_repetition = (None, 0)

            while incomplete_moves and time() < phase_end_time:

                # On each iteration, choose a move to evaluate using a similar scheme to UCT
                exploration_const = np.sqrt(np.log(num_evals_done + 1)) * self.move_config.sampling_exploration_coef
                values = {
                    move: np.inf if move not in move_scores or move_scores[move].num_samples == 0 else (
                        exploration_const * np.sqrt(1 / move_scores[move].num_samples)
                        + move_scores[move].minimum * self.move_config.min_score_factor
                        + move_scores[move].maximum * self.move_config.max_score_factor
                        + move_scores[move].average * self.move_config.mean_score_factor
                    ) for move in moves
                }
                # First evaluate current best move for possible early stopping
                top_move = max(moves, key=values.get)
                prev_top_move, num_reps = top_move_repetition
                if top_move == prev_top_move:
                    top_move_repetition = (top_move, num_reps + 1)
                    if num_reps >= self.move_config.move_sample_rep_limit and top_move not in incomplete_moves:
                        self.logger.debug("Move choice seems to be converged; breaking loop")
                        break
                else:
                    top_move_repetition = (top_move, 0)
                # Otherwise, sample a move to evaluate
                move_to_eval = max(incomplete_moves, key=values.get)

                # Then iterate through boards in descending priority to get one for eval
                needed_boards_for_eval = boards_to_sample[move_to_eval]
                for priority in sorted_priorities:
                    priority_boards = needed_boards_for_eval & self.board_sample_priority[priority]
                    if priority_boards:
                        board_to_eval = priority_boards.pop()

                        # Get the score for the corresponding taken move, then map back to all equivalent move requests
                        taken_move_to_eval = all_maps_to_taken_move[board_to_eval][move_to_eval]

                        # Get the board position score before moving
                        score_before_move, _ = self.memo_calc_score(board_to_eval, key=make_cache_key(board_to_eval))

                        # Get the score and update the estimate
                        score, _ = self.memo_calc_score(
                            board_to_eval, taken_move_to_eval, -score_before_move,
                            make_cache_key(board_to_eval, taken_move_to_eval, -score_before_move),
                        )
                        weight = self.weight_board_probability(score_before_move)
                        for requested_move in all_maps_from_taken_move[board_to_eval][taken_move_to_eval]:
                            move_scores[requested_move].update(score, weight)

                            boards_to_sample[requested_move].remove(board_to_eval)
                            if not boards_to_sample[requested_move]:
                                incomplete_moves.remove(requested_move)

                            num_evals_done += 1
                            pbar.update()

                        break

                else:
                    raise AssertionError("This can only be reached if a move eval is requested when already completed")

                pbar.update()

        self.logger.debug(f"Had {num_precomputed} of {total_evals} already cached")
        self.logger.debug(f"Spent {time() - start_time:0.1f} seconds computing new move scores")
        self.logger.debug(f"Sampled {num_evals_done} of {total_evals} move+board pairs.")

        # Combine the mean, min, and max possible scores based on config settings
        compound_score = {
            move: (
                    est.minimum * self.move_config.min_score_factor
                    + est.maximum * self.move_config.max_score_factor
                    + est.average * self.move_config.mean_score_factor
            ) for move, est in move_scores.items()
        }

        # Determine the minimum score a move needs to be considered
        highest_score = max(compound_score.values())
        threshold_score = highest_score - self.move_config.threshold_score

        self.logger.debug("Move values presented as: (min, mean, max, compound) @ samples")
        for move, est in move_scores.items():
            self.logger.debug(f"{move.uci()}: ("
                              f"{est.minimum:5.0f}, "
                              f"{est.average:5.0f}, "
                              f"{est.maximum: 5.0f}, "
                              f"{compound_score[move]: 5.0f})"
                              f" @ {est.num_samples}")

        # Create a list of all moves which scored above the threshold
        move_options = [move for move, score in compound_score.items() if score >= threshold_score]
        # Randomly choose one of the remaining moves
        move_choice = random.choice(move_options)

        return move_choice

    def _get_engine_move(self, board: chess.Board):
        self.logger.debug("Only one possible board; using win-or-engine-move method")

        # Capture the opponent's king if possible
        if board.was_into_check():
            op_king_square = board.king(not board.turn)
            king_capture_moves = [
                move for move in board.pseudo_legal_moves
                if move and move.to_square == op_king_square
            ]
            return random.choice(king_capture_moves)

        # If in checkmate or stalemate, return
        if not board.legal_moves:
            return PASS

        # Otherwise, let the engine decide

        # For Stockfish versions based on NNUE, seem to need to screen illegal en passant
        replaced_ep_square = None
        if board.ep_square is not None and not board.has_legal_en_passant():
            replaced_ep_square, board.ep_square = board.ep_square, replaced_ep_square
        # Then analyse the position
        move = self.engine.play(board, limit=chess.engine.Limit(time=2.0)).move
        # Then put back the replaced ep square if needed
        if replaced_ep_square is not None:
            board.ep_square = replaced_ep_square

        self.logger.debug(f"Engine chose to play {move}")
        return move

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        super().handle_move_result(requested_move, taken_move, captured_opponent_piece, capture_square)
        if self.opening_book is not None:
            move_result = (taken_move, capture_square)
            move_result_str = f"({'null' if taken_move is None else taken_move.uci()}," \
                              f" {'-' if capture_square is None else chess.SQUARE_NAMES[capture_square]})"
            if move_result in self.opening_book:
                self.logger.debug(f"Move result {move_result_str} is in opening book")
                self.opening_book = self.opening_book[move_result]
            else:
                self.logger.debug(f"Move result {move_result_str} is not in opening book")
                self.opening_book = None

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
                self.logger.debug(f"Requested board scores when king was missing! {uncached_board.epd()}")

        # Otherwise, calculate move scores for a random board that could be reached in two turns
        elif self.while_we_wait_extension and self.next_turn_boards_unsorted:
            board = fast_copy_board(random.choice(tuple(self.next_turn_boards_unsorted)))
            board.push(random.choice(rbc_legal_moves(board)))
            board.push(random.choice(rbc_legal_moves(board)))
            if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
                self.cache_board(board)

        else:
            sleep(0.001)

    def gameover_strategy(self):
        """
        Quit the StockFish engine instance(s) associated with this strategy once the game is over.
        """
        self.logger.debug(
            f"During this game, scored {len(self.score_calc_times)} positions "
            f"at average {self.calc_time_per_move():.5f} seconds per score "
            f"using search depth {self.score_config.search_depth}."
            f" Engine cache had {ENGINE_CACHE_STATS[0]} hits and {ENGINE_CACHE_STATS[1]} misses."
        )

        # Shut down StockFish
        self.logger.debug("Terminating engine.")
        self.engine.quit()
        self.logger.debug("Engine exited.")
