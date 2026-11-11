import os
import sys
import pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
out, smo, sno = sys.argv[1:4]
def grp(s):
    s = str(s); return "agree" if s.startswith("agree") else ("dis" if s.startswith("dis") else s)
m = pd.read_csv(f"{R}/submissions/{smo}_detail.csv").set_index("Unnamed: 0"); n = pd.read_csv(f"{R}/submissions/{sno}_detail.csv").set_index("Unnamed: 0")
om, on = m.answer_source.map(grp), n.answer_source.map(grp).reindex(m.index)
sel = ((on == "agree") & (om == "dis")) | ((on == "geo_cawinvalid") & (om == "h1"))
o = m.copy(); o.loc[sel, "answer"] = n.answer.reindex(m.index)[sel]; o.loc[sel, "answer_source"] = "mo_guard:" + n.answer_source.reindex(m.index)[sel].astype(str)
sub = pd.read_csv(f"{R}/submissions/{smo}_id1.csv")
sub["parsed_value"] = o.answer.values
sub.to_csv(f"{R}/submissions/{out}_id1.csv", index=False)
o.to_csv(f"{R}/submissions/{out}_detail.csv")
