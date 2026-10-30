"""Point-in-time stat projections from fantasy.py's own backtest(), imported read-only.
Data and outputs go to this scratch folder; nothing in /mnt/project-files/fantasy is written."""
import sys, os, pickle
sys.path.insert(0, "/mnt/project-files/fantasy")
import fantasy as F
here = os.path.dirname(os.path.abspath(__file__))
F.DATA, F.OUT = os.path.join(here, "fdata"), os.path.join(here, "fout")
if "--no-download" not in sys.argv: F.download()
d = F.load()
prep = F.prepare(d)
weeks = [(2025, w) for w in range(4, 19)] + [(2026, w) for w in range(3, 9)]
b = F.backtest(d, prepared=prep, quiet=True, weeks=weeks)
b.to_pickle(os.path.join(here, "proj_bt.pkl"))
print(b.shape, b.columns.tolist())
