"""Run all experiments listed in ett_exp.md and log output."""

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_commands(md_path: Path):
    commands = []
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("python "):
            commands.append(line)
    return commands


def run_command(command, log_fp):
    log_fp.write("\n$ " + command + "\n")
    log_fp.flush()
    print("\n$ " + command)
    args = shlex.split(command, posix=False)
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_fp.write(line)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", default="ett_exp.md", help="Markdown file with commands.")
    parser.add_argument("--log-dir", default="logs", help="Directory to write log files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands.")
    args = parser.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        raise FileNotFoundError(f"Missing markdown file: {md_path}")

    commands = parse_commands(md_path)
    if not commands:
        print("No commands found.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ett_exp_run_{timestamp}.log"

    if args.dry_run:
        print("Commands to run:")
        for cmd in commands:
            print(cmd)
        return

    print("Logging to:", log_path)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_fp:
        log_fp.write(f"Start: {timestamp}\n")
        log_fp.write(f"Source: {md_path}\n")
        log_fp.write(f"Total commands: {len(commands)}\n")
        for command in commands:
            run_command(command, log_fp)


if __name__ == "__main__":
    main()
