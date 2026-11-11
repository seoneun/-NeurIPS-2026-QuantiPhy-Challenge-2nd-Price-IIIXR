import os, pickle
ROWS = pickle.load(open(f"{os.environ['QP_WORK']}/h/rows.pkl", "rb"))["rows"]
