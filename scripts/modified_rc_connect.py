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

import json
from datetime import datetime
import logging
import multiprocessing
import signal
import sys
import time
import traceback
import click
import requests

import chess
from reconchess import Player, RemoteGame, play_turn, ChessJSONDecoder, ChessJSONEncoder
from reconchess.scripts.rc_connect import RBCServer, check_package_version

from strangefish.strangefish_strategy import StrangeFish2
from strangefish.utilities import ignore_one_term
from strangefish.utilities.player_logging import create_file_handler, create_stream_handler


class OurRemoteGame(RemoteGame):

    def __init__(self, server_url, game_id, auth):
        self.logger = logging.getLogger(f"game-{game_id}.server-comms")
        super().__init__(server_url, game_id, auth)

    def is_op_turn(self):
        status = self._get("game_status")
        return not status["is_over"] and not status["is_my_turn"]

    def _get(self, endpoint, decoder_cls=ChessJSONDecoder):
        self.logger.debug(f"Getting '{endpoint}'")
        done = False
        url = "{}/{}".format(self.game_url, endpoint)
        while not done:
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    done = True
                elif response.status_code >= 500:
                    time.sleep(0.5)
                else:
                    self.logger.error(response.text)
                    raise ValueError(response.text)
            except requests.RequestException as e:
                self.logger.error(e)
                time.sleep(0.5)
        response = response.json(cls=decoder_cls)
        self.logger.debug(f"response: {response}")
        return response

    def _post(self, endpoint, obj):
        self.logger.debug(f"Posting '{endpoint}' -> {obj}")
        url = "{}/{}".format(self.game_url, endpoint)
        data = json.dumps(obj, cls=ChessJSONEncoder)
        done = False
        while not done:
            try:
                response = self.session.post(url, data=data)
                if response.status_code == 200:
                    done = True
                elif response.status_code >= 500:
                    time.sleep(0.5)
                else:
                    self.logger.error(response.text)
                    raise ValueError(response.text)
            except requests.RequestException as e:
                self.logger.error(e)
                time.sleep(0.5)
        response = response.json(cls=ChessJSONDecoder)
        self.logger.debug(f"response: {response}")
        return response


def our_play_remote_game(server_url, game_id, auth, player: Player):
    game = OurRemoteGame(server_url, game_id, auth)
    logger = logging.getLogger(f"game-{game_id}.game-mod")

    op_name = game.get_opponent_name()
    our_color = game.get_player_color()

    logger.debug("Setting up remote game %d playing %s against %s.",
                 game_id, chess.COLOR_NAMES[our_color], op_name)

    player.handle_game_start(our_color, game.get_starting_board(), op_name)
    game.start()

    turn_num = 0
    while not game.is_over():
        turn_num += 1
        logger.info("Playing turn %2d. (%3.0f seconds left.)", turn_num, game.get_seconds_left())
        play_turn(game, player, end_turn_last=False)
        logger.info("   Done turn %2d. (%d boards.)", turn_num, len(player.boards))

        if hasattr(player, "while_we_wait") and getattr(player, "while_we_wait"):
            while game.is_op_turn():
                player.while_we_wait()

    winner_color = game.get_winner_color()
    win_reason = game.get_win_reason()
    game_history = game.get_game_history()

    logger.debug("Ending remote game %d against %s.", game_id, op_name)

    player.handle_game_end(winner_color, win_reason, game_history)

    return winner_color, win_reason, game_history


def accept_invitation_and_play(server_url, auth, invitation_id, finished):
    # make sure this process doesn't react to the first interrupt signal
    signal.signal(signal.SIGINT, ignore_one_term)

    logger = logging.getLogger("rc-connect")

    logger.debug("Accepting invitation %d.", invitation_id)
    server = RBCServer(server_url, auth)
    game_id = server.accept_invitation(invitation_id)
    logger.info("Invitation %d accepted. Playing game %d.", invitation_id, game_id)

    player = StrangeFish2(game_id=game_id)

    try:
        our_play_remote_game(server_url, game_id, auth, player)
        logger.debug("Finished game %d.", game_id)
    except:
        logging.getLogger(f"game-{game_id}").exception("Fatal error in game %d.", game_id)
        traceback.print_exc()
        server.error_resign(game_id)
        player.handle_game_end(None, None, None)
        logger.critical("Game %d closed on account of error.", game_id)
    finally:
        server.finish_invitation(invitation_id)
        finished.value = True
        logger.debug("Game %d ended. Invitation %d closed.", game_id, invitation_id)


def listen_for_invitations(server, max_concurrent_games, limit_games):
    logger = logging.getLogger("rc-connect")

    connected = False
    process_by_invitation = {}
    finished_by_invitation = {}
    num_games_joined = 0
    disconnect_signalled = False
    while True:
        try:
            # get unaccepted invitations
            invitations = server.get_invitations()

            # set max games on server if this is the first successful connection after being disconnected
            if not connected:
                logger.info("Connected successfully to server!")
                connected = True
                server.set_max_games(max_concurrent_games)

            # filter out finished processes
            finished_invitations = []
            for invitation in process_by_invitation.keys():
                if not process_by_invitation[invitation].is_alive() or finished_by_invitation[invitation].value:
                    finished_invitations.append(invitation)
            for invitation in finished_invitations:
                logger.info(f"Terminating process for invitation {invitation}"
                            f" (exit code: {process_by_invitation[invitation].exitcode})")
                process_by_invitation[invitation].terminate()
                del process_by_invitation[invitation]
                del finished_by_invitation[invitation]

            # Optionally, disconnect after N games joined, and exit the script when all game completed
            if limit_games is not None and num_games_joined >= limit_games:
                if not disconnect_signalled:
                    server.set_max_games(0)
                    server.set_ranked(False)
                    disconnect_signalled = True
                elif not process_by_invitation and not invitations:
                    return

            # accept invitations until we have #max_concurrent_games processes alive
            for invitation in invitations:
                # only accept the invitation if we have room and the invite doesn't have a process already
                if invitation not in process_by_invitation:
                    logger.debug(f"Received invitation {invitation}.")

                    if len(process_by_invitation) < max_concurrent_games:
                        # start the process for playing a game
                        finished = multiprocessing.Value("b", False)
                        process = multiprocessing.Process(
                            target=accept_invitation_and_play,
                            args=(server.server_url, server.session.auth, invitation, finished))
                        process.start()
                        num_games_joined += 1

                        # store the process so we can check when it finishes
                        process_by_invitation[invitation] = process
                        finished_by_invitation[invitation] = finished
                    else:
                        logger.info(f"Not enough game slots to play invitation {invitation}.")
                        server.set_ranked(False)
                        max_concurrent_games += 1

        except requests.RequestException as e:
            connected = False
            logger.exception("Failed to connect to server")
            print(e)
        except Exception:
            logger.exception("Error in invitation processing: ")
            traceback.print_exc()

        time.sleep(5)


@click.command()
@click.argument("username")
@click.argument("password")
@click.option("--server-url", "server_url", default="https://rbc.jhuapl.edu", help="URL of the server.")
@click.option("--max-concurrent-games", "max_concurrent_games", type=int, default=1, help="Maximum games to play at once.")
@click.option("--limit-games", "limit_games", type=int, default=None, help="Optional limit to number of games played.")
@click.option("--ranked", "ranked", type=bool, default=False, help="Play for leaderboard ELO.")
@click.option("--keep-version", "keep_version", type=bool, default=True, help="Keep existing leaderboard version num.")
def main(username, password, server_url, max_concurrent_games, limit_games, ranked, keep_version):

    logger = logging.getLogger("rc-connect")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(create_stream_handler())
    logger.addHandler(create_file_handler(f"rc_connect_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"))

    logger.debug(
        f"Running modified_rc_connect to play RBC games online."
        f" {server_url=}"
        f" {max_concurrent_games=}"
        f" {limit_games=}"
        f" {ranked=}"
        f" {keep_version=}"
    )

    auth = username, password
    server = RBCServer(server_url, auth)

    # verify we have the correct version of reconchess package
    check_package_version(server)

    def handle_term(signum, frame):
        # reset to default response to interrupt signals
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        logger.warning("Received terminate signal, waiting for games to finish and then exiting.")
        server.set_ranked(False)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_term)

    # tell the server whether we want to do ranked matches or not
    if ranked:
        if not keep_version:
            server.increment_version()
        server.set_ranked(True)
    else:
        server.set_ranked(False)

    listen_for_invitations(server, max_concurrent_games, limit_games)


if __name__ == "__main__":
    main()
