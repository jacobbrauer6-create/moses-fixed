"""CLI entry point for moses benchmarking."""
import argparse

def main():
    parser = argparse.ArgumentParser(description="MOSES benchmarking platform")
    parser.add_argument("command", choices=["train","sample","metrics"],
                        help="Command to run")
    args = parser.parse_args()
    print(f"moses {args.command} — see documentation at https://github.com/molecularsets/moses")

if __name__ == "__main__":
    main()
