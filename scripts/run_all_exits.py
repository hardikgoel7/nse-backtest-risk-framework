"""One command: runs every pre-declared exit variant on Nifty 50 (and Nifty 500 if its CSV exists)
and saves ALL output to reports/all_runs.txt (attach that file in chat).
Variants: +30% [primary, pre-declared], +20% [exploratory], 3R [exploratory], RSI>=60 [original exit rule].
Usage: python scripts/run_all_exits.py"""
import os, subprocess, sys, time

variants = [("t30", ["--target", "0.30"]), ("t20", ["--target", "0.20"]),
            ("r3", ["--rmult", "3"]), ("rsi60", ["--exit-rsi", "60"])]
universes = [("n50", [])]
if os.path.exists("data/nifty500_constituents.csv"):
    universes.append(("n500", ["--universe", "data/nifty500_constituents.csv"]))
else:
    print("NOTE: data/nifty500_constituents.csv not found -> Nifty 50 only")

os.makedirs("reports", exist_ok=True)
with open("reports/all_runs.txt", "w", encoding="utf-8") as out:
    for u, uargs in universes:
        for v, vargs in variants:
            tag = f"{u}_{v}"; t0 = time.time()
            print(f"running {tag} ...", flush=True)
            r = subprocess.run([sys.executable, "scripts/run_exit_rules.py", *uargs, *vargs, "--tag", tag],
                               capture_output=True, text=True)
            out.write(f"\n\n######## {tag} ########\n{r.stdout}\n{r.stderr[-1500:] if r.returncode else ''}")
            print(f"  done in {time.time() - t0:.0f}s" + ("  FAILED, see file" if r.returncode else ""))
print("\nAll done. Attach reports/all_runs.txt in chat.")
