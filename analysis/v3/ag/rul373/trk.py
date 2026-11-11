import json, sys, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def load(video):
    return json.load(open(os.path.join(R, 'geo/out/tracks_test', video + '.json')))

def series(d, phrase):
    for q in d['queries']:
        if phrase in q['phrase']:
            out = []
            for i, fr in enumerate(q['frames']):
                st = fr.get('st') if isinstance(fr, dict) else None
                if st and st.get('box'):
                    out.append((d['fidx'][i], st['box'], st.get('c'), st.get('rr'), st.get('area'), st.get('border')))
            return q, out
    return None, []

if __name__ == '__main__':
    v = sys.argv[1]; step = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    ph = sys.argv[3] if len(sys.argv) > 3 else None
    d = load(v)
    print(v, 'N', d['N'], 'fps', d['fps'], 'W', d['W'], 'H', d['H'])
    for q in d['queries']:
        if ph and ph not in q['phrase']:
            continue
        _, s = series(d, q['phrase'])
        print('')
        for (f, b, c, rr, a, bo) in s[::step]:
            print('')


def objseries(video, phrase, dirs=('tracks_test_v2sc', 'tracks_test_v3ntmv')):
    for dd in dirs:
        p = os.path.join(R, 'geo/out', dd, video + '.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for q in d['queries']:
            if phrase in q.get('phrase', '') and q.get('obj') is not None and str(q['obj']) in d['objs']:
                fr = d['objs'][str(q['obj'])]
                out = [(d['fidx'][i], s['st']) for i, s in enumerate(fr) if isinstance(s, dict) and isinstance(s.get('st'), dict) and s['st'].get('box')]
                return dd, d, out
    return None, None, []
