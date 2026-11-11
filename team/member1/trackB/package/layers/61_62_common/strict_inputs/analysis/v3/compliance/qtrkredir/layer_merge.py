import os, sys, json
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
LAY = {int(r): int(g["gid"].split("__q")[1])
       for g in json.load(open(f"{R}/analysis/v3/compliance/qtrack/out/groups_test.json"))["groups"] for r in g["rows"]}


def patches(wd, nl, tags):
    for t in tags:
        out, kept, other = {}, [0] * nl, 0
        for k in range(nl):
            p = f"{wd}/L{k}/patches/ag_{t}_patch_test.json"
            if not os.path.exists(p):
                continue
            for key, e in json.load(open(p)).items():
                r = int(key); assert r in LAY
                if LAY[r] == k:
                    out[key] = e; kept[k] += 1
                else:
                    other += 1
        out = {k_: out[k_] for k_ in sorted(out, key=int)}
        json.dump(out, open(f"{R}/cache/tmp/ag_{t}_patch_test.json", "w"), indent=0)
        print('')


if __name__ == "__main__":
    patches(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
