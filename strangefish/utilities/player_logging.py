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

    Based on https://github.com/ginop/reconchess-strangefish
    Original copyright (c) 2019, Gino Perrotta, Robert Perrotta, Taylor Myers
"""

import logging
import logging.handlers
import sys
import os

TAGS = [
    "\N{four leaf clover}",
    "\N{skull}",
    "\N{bacon}",
    "\N{spouting whale}",
    "\N{fire}",
    "\N{eagle}",
    "\N{smiling face with sunglasses}",
    "\N{beer mug}",
    "\N{rocket}",
    "\N{snake}",
    "\N{butterfly}",
    "\N{jack-o-lantern}",
    "\N{white medium star}",
    "\N{hot beverage}",
    "\N{earth globe americas}",
    "\N{red apple}",
    "\N{robot face}",
    "\N{sunflower}",
    "\N{doughnut}",
    "\N{crab}",
    "\N{soccer ball}",
    "\N{hibiscus}",
]

concise_format = "%(process)-5d %(asctime)8s  {}  %(message)s"
verbose_format = logging.Formatter(
    "%(name)s %(levelname)s: \"%(message)s\" from %(module)s.%(funcName)s line %(lineno)d at %(asctime)s"
)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "game_logs")
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.exists(os.path.join(LOG_DIR, "engine_logs")):
    os.mkdir(os.path.join(LOG_DIR, "engine_logs"))


def create_stream_handler(log_level=logging.INFO):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(logging.Formatter(concise_format.format(TAGS[os.getpid() % len(TAGS)]), "%I:%M:%S"))
    return stdout_handler


def create_file_handler(filename: str, max_bytes: int = None):
    if max_bytes is None:
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, filename), mode="w")
    else:
        file_handler = logging.handlers.RotatingFileHandler(os.path.join(LOG_DIR, filename), "a", max_bytes, 1)
    file_handler.setFormatter(verbose_format)
    file_handler.setLevel(logging.DEBUG)
    return file_handler
