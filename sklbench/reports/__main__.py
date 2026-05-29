# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import sys

from .speedups import add_speedups_parser, build_speedups_report


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="python -m sklbench.reports")
    subparsers = parser.add_subparsers(dest="command", required=True)
    speedups_parser = subparsers.add_parser(
        "speedups",
        help="Build an interactive HTML speed-up report.",
    )
    add_speedups_parser(speedups_parser)

    args = parser.parse_args()
    try:
        if args.command == "speedups":
            return build_speedups_report(args)
        parser.error(f"Unknown command: {args.command}")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
