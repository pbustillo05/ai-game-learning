"""Replica-only production queued benchmark consumer."""

from pbg.cli.benchmark import run_cli


def main() -> None:
    run_cli(require_replica=True)


if __name__ == "__main__":
    main()
