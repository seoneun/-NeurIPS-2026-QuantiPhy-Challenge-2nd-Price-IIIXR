import csv, io, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LINE_END = {'lf': '\n', 'crlf': '\r\n'}


def main():
    out, names = os.path.abspath(sys.argv[1]), sys.argv[2:]
    with open(os.path.join(HERE, 'steps', 'MANIFEST.csv'), encoding='utf-8', newline='') as f:
        manifest = {r['step']: r for r in csv.DictReader(f)}
    for name in names:
        st = manifest[name]
        with open(os.path.join(out, st['input']), encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
        k, ki = rows[0].index('parsed_value'), rows[0].index('id')
        pos = {int(r[ki]) - 1: n for n, r in enumerate(rows) if n > 0}
        with open(os.path.join(HERE, st['table']), encoding='utf-8', newline='') as f:
            for t in csv.DictReader(f):
                rows[pos[int(t['id0'])]][k] = t['value']
        s = io.StringIO(newline='')
        csv.writer(s, lineterminator=LINE_END[st['line_end']]).writerows(rows)
        open(os.path.join(out, st['output']), 'wb').write(s.getvalue().encode('utf-8'))


if __name__ == '__main__':
    main()
