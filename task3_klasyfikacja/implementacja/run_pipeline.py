import argparse
import subprocess
import sys
import os


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spacer.csv")
    ap.add_argument("--target", default="Spacer")
    ap.add_argument("--algo", choices=["id3", "c45", "both"], default="both")
    args = ap.parse_args()

    py = sys.executable

    if not os.path.exists(args.data):
        print(f"Error: File {args.data} not found.")
        sys.exit(1)

    if args.algo in ["id3", "both"]:
        run([py, "01_id3.py", "--data", args.data, "--target", args.target])

    if args.algo in ["c45", "both"]:
        run([py, "02_c45.py", "--data", args.data, "--target", args.target])


if __name__ == "__main__":
    main()