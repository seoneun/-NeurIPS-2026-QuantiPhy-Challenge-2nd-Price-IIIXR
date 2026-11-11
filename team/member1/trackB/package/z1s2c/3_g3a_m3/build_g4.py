import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g4io as io
_N = {'c2': 'pred_t015', 'c3a': 'pred_t115', 'c3b': 'pred_t116', 'c3m': 'pred_t117', 'c4': 'pred_t118', 'c4m': 'pred_t119', 'c4s': 'pred_t120', 'c5': 'pred_t121', 'c5m': 'pred_t021', 'c6': 'pred_t122', 'c7': 'pred_t123', 'c8': 'pred_t124', 'c9': 'pred_t125', 'c9b': 'pred_t126', 'c9n': 'pred_t127', 'c9u': 'pred_t128', 'g2': 'pred_t129', 'g2a': 'pred_t130', 'g3': 'pred_t131', 'g3a': 'pred_t132', 'i1': 'pred_t133', 'j1': 'pred_t134', 'j6': 'pred_t135', 'j7': 'pred_t136', 'j7s': 'pred_t137', 'j9n': 'pred_t138', 'm3': 'pred_t139', 'p1': 'pred_t140', 'r1': 'pred_t141', 's1': 'pred_t142', 'y1': 'pred_t143', 'y1s': 'pred_t144', 'z1': 'pred_t145', 'z1s2': 'pred_t146', 'z1s2c': 'pred_t004'}

G, G3A, C9U, C9N, OUT = sys.argv[1:6]
pp = lambda f: os.path.join(G, "inputs", "patches", f)
o = lambda n: os.path.join(OUT, f"{_N[n]}_id1.csv")
log = {}

P = io.load_patch(pp("r40_i1_test.json")); io.write_lines(G3A, P, o("i1")); log["i1"] = len(P)

P = io.load_patch(pp("r41_PHYSFIT_test.json")); io.write_float(io.read_float(o("i1")), P, o("j1")); log["j1"] = len(P)

F3 = io.load_patch(pp("r46_F3_test.json")); S = io.load_patch(pp("r47_BUILD_S_test.json"))
B = dict(F3); B.update(S)
io.write_float(io.read_float(o("j1")), B, o("j6")); log["j6"] = len(B)

P = io.load_patch(pp("r48_BUILD_WSt_test.json")); io.write_float(io.read_float(o("j6")), P, o("j7")); log["j7"] = len(P)

J7 = io.read_float(o("j7")); CU = io.read_float(C9U); cat = io.cat_of(J7)
P = {i: float(CU.parsed_value[i]) for i in range(len(J7)) if cat[i] == "S2" and J7.parsed_value[i] != CU.parsed_value[i]}
io.write_float(J7, P, o("j7s")); log["j7s"] = len(P)

J7S = io.read_float(o("j7s")); CN = io.read_float(C9N); cat = io.cat_of(J7S)
P = {i: float(CN.parsed_value[i]) for i in range(len(J7S))
     if cat[i] == "S3" and J7S.parsed_value[i] == CU.parsed_value[i] and CN.parsed_value[i] != CU.parsed_value[i]}
M = io.load_patch(pp("port2_M_test.json"))
P[500] = M[500]
io.write_float(J7S, P, o("j9n")); log["j9n"] = len(P)

J9N = io.read_float(o("j9n")); cat = io.cat_of(J9N)
P = {k: v for k, v in io.load_patch(pp("finalM_j9n_GMf_test.json")).items() if cat[k].startswith("D")}
io.write_str(o("j9n"), P, o("m3")); log["m3"] = len(P)

print("ROWS_CHANGED", json.dumps(log))
