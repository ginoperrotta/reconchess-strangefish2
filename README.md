# StrangeFish2

StrangeFish2 is a bot created to play Reconnaissance Blind Chess (RBC).
It won the NeurIPS 2022 RBC tournament
and placed second in the NeurIPS 2021 RBC tournament!
For more information about RBC and the tournaments, see
https://rbc.jhuapl.edu/ and https://rbc.jhuapl.edu/past_competitions.

StrangeFish2 is a continuation of the NeurIPS 2019 RBC tournament 
winner, StrangeFish. https://github.com/ginop/reconchess-strangefish. 
StrangeFish2 has a similar architecture, but is much more efficient 
compared to the previous implementation. More critically, the sense 
strategy has been modified and is much stronger.

## Overview

StrangeFish is separated into two parts: the primary component
manages game-state information throughout the game,
and the secondary component makes decisions for sensing and
moving based on that available information.
Because of this separation, it should be simple for users to
implement their own strategies to use within StrangeFish.

StrangeFish maintains a set of all possible board states
by expanding each possible board
into a new set for each possible opponent's move each turn.
The set of possible board states is reduced by
comparing hypothetical observations to in-game results for
sensing, moving, and the opponent's move's capture square.
Expansion and filtering of the board set are parallelized,
and set expansion is begun during the opponent's turn
(when possible).

The sensing strategy does calculate and consider the expected
board set size reduction, but the primary influence on
sensing decisions is the expected outcome of the following
move choice. Before sensing, the bot calculates scores for each 
move option on each possible board (or for as large of a random 
sample as time allows). It also computes subsets of those boards which
would be validated by each possible sense observation and estimates the 
likelihood of each observation. It then chooses moves for each of those 
possibilities and computes the expected result of that move based on 
the estimated probabilities of the boards in that subset.
The sense choice is the square with the greatest expected outcome averaged
over possible observations, move choices, and move results.

The sensing decision is therefore coupled with the move strategy.
In this case, that strategy is a flexible combination
of the best-case, worst-case, and expected (weighted-average)
outcomes for each available move across the board set.
The relative contributions of the mean, min, and max scores
are tunable. The scores for each individual move
were calculated using StockFish when possible, with a set
of heuristics for evaluating board states unique to RBC
such as staying or moving into check or actually capturing
the opponent's king. The moving strategy also contributes
to the board set size maintenance by awarding additional
points to moves which are expected to eliminate possibilities
(for example, a sliding move which may travel through
potentially-occupied squares).

## Setup

We use conda to manage our execution environment.
```
conda env create --file environment.yml
conda activate strangefish
export STOCKFISH_EXECUTABLE=/absolute/path/to/stockfish
export PYTHONPATH='.':$PYTHONPATH
```

## Local bot matches

See `scripts/example_game.py` 
to play a local RBC game using StrangeFish2.

## Server matches

To connect to the server as was done for the NeurIPS 2021 tournament,
use `scripts/modified_rc_connect.py`, implemented here with
click command-line arguments.

Ex: `python scripts/modified_rc_connect.py [your username] [your password] --max-concurrent-games 1`
