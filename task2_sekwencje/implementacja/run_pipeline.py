import argparse
import subprocess
import sys
import os

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["small", "expanded"], default="small")
    ap.add_argument("--min-sup-count", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--min-lift", type=float, default=None)
    ap.add_argument("--algo", choices=["apriori", "fpgrowth"], default="apriori")
    ap.add_argument("--task", choices=["basket", "sequence"], default="basket")
    ap.add_argument("--seq-algo", choices=["apriori_all", "spade"], default="apriori_all")
    ap.add_argument("--n-transactions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-items", type=int, default=5000)
    ap.add_argument("--target-transactions", type=int, default=50000)
    ap.add_argument("--expanded-seed", type=int, default=123)
    ap.add_argument("--sequences", default=None)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-apriori", action="store_true")
    ap.add_argument("--skip-rules", action="store_true")

    args = ap.parse_args()
    py = sys.executable

    if not args.skip_generate:
        if args.dataset == "small":
            run([py, "01_generate_data.py",
                 "--n-transactions", str(args.n_transactions),
                 "--seed", str(args.seed),
                 "--target-items", "22",
                 "--tail-prob", "0.0"])
        else:
            run([py, "01_generate_data.py",
                 "--n-transactions", str(args.target_transactions),
                 "--seed", str(args.expanded_seed),
                 "--target-items", str(args.target_items),
                 "--tail-prob", "0.10",
                 "--basket-min", "3",
                 "--basket-max", "9"])

    if args.task == "sequence" and args.seq_algo == "spade" and args.sequences:
        if args.sequences.endswith(".txt"):
            target_csv = args.sequences.replace(".txt", ".csv")
            if not os.path.exists(target_csv):
                print(f"--- Konwersja: {args.sequences} -> {target_csv} ---")
                run([py, "00_convert_zaki.py", "--input", args.sequences, "--out", target_csv])
            args.sequences = target_csv

    if not args.skip_apriori:
        if args.task == "basket":
            if args.algo == "apriori":
                cmd = [py, "02_apriori.py"]
            else:
                cmd = [py, "02_fpgrowth.py"]
        else:
            if args.seq_algo == "apriori_all":
                cmd = [py, "02_apriori_all.py"]
            else:
                cmd = [py, "02_spade.py"]

        if args.min_sup_count is not None:
            cmd += ["--min-sup-count", str(args.min_sup_count)]
        if args.task == "sequence" and args.sequences is not None:
            cmd += ["--sequences", args.sequences]

        run(cmd)

    if not args.skip_rules and args.task == "basket":
        cmd = [py, "03_generate_rules.py"]
        if args.min_conf is not None:
            cmd += ["--min-conf", str(args.min_conf)]
        if args.min_lift is not None:
            cmd += ["--min-lift", str(args.min_lift)]
        run(cmd)

    print("OK: pipeline zakończony. Sprawdź out/")

if __name__ == "__main__":
    main()

r"""
# =============================================================================
# PRZYKŁADY URUCHOMIENIA DLA SEKWENCJI
# =============================================================================

# 1. DANE GENEROWANE (SMALL)
# AprioriAll
python run_pipeline.py --task sequence --seq-algo apriori_all --min-sup-count 8 --skip-rules
# SPADE
python run_pipeline.py --task sequence --seq-algo spade --min-sup-count 8 --skip-rules
c
# 3. ZBIÓR DELICIOUS (DUŻE DANE)
python run_pipeline.py --task sequence --seq-algo spade --sequences data/raw/delicious_sequence.txt --min-sup-count 100 --skip-generate
# Dokładne odwzorowanie PDF (wsparcie ~0.2%, może chwilę potrwać):
python run_pipeline.py --task sequence --seq-algo spade --sequences data/raw/delicious_sequence.txt --min-sup-count 10 --skip-generate

# 4. PORÓWNANIE ALGORYTMÓW NA TYCH SAMYCH DANYCH
python run_pipeline.py --task sequence --seq-algo apriori_all --sequences data/spade_test.csv --min-sup-count 2 --skip-generate
python run_pipeline.py --task sequence --seq-algo spade --sequences data/spade_test.csv --min-sup-count 2 --skip-generate
"""