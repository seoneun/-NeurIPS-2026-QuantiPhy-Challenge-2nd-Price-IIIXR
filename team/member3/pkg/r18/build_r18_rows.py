import os
import sys
import pandas as pd

O = sys.argv[1]
os.makedirs(O, exist_ok=True)
TABLES = {
    'h74D2': {1840: 4.6, 1838: 5.0, 645: 1.22, 1971: 2.1, 1975: 2.5, 1974: 1.57, 796: 1.55, 1990: 1.96},
    'h78S2': {347: 1.85, 716: 0.48, 725: 0.48, 717: 2.3},
    'h78D2': {1844: 0.36, 565: 0.51, 566: 0.68, 567: 0.505, 568: 1.42, 569: 0.59, 578: 1.40, 612: 0.249, 613: 0.50, 614: 0.672,
              616: 1.42, 617: 1.44, 618: 0.947, 620: 1.80, 580: 1.04, 581: 0.875, 582: 0.90, 697: 1.49, 708: 1.63, 710: 1.56},
}
for name, rows in TABLES.items():
    pd.DataFrame(sorted(rows.items()), columns=['id0', 'value']).to_csv(f'{O}/{name}_test.csv', index=False)
