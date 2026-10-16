# Table B (compact). Frozen-route characteristics

Frozen customer-route characteristics for TRAIN, VALIDATION, and the V4 TEST corpus. Mean customer-only distance is the mean length of the frozen depot–customers–depot sequence in normalized SynthCharge coordinate units. Instance coordinates lie in the generator's unit square. These distances are not kilometers. Charging-required and no-charge-required counts are quota labels. Certificate station visits count CHARGE steps in the stored witness, not FA-HPPO decisions. Layouts are clustered (C), random (R), and mixed (RC). Length groups are short (3–5), medium (6–10), and long (11–15) customers on the frozen route.

| Split | Routes | Customers mean [min, median, max] | Mean distance (coordinate units) | Charging / no charge | C / R / RC | Short / medium / long | Certificate visits mean [min, median, max] |
|---|---:|---|---:|---|---|---|---|
| TRAIN | 180 | 8.68 [3, 8.5, 15] | 1.744 | 135 / 45 | 60 / 60 / 60 | 60 / 60 / 60 | 0.77 [0, 1.0, 3] |
| VALIDATION | 90 | 8.50 [4, 8.5, 15] | 1.820 | 72 / 18 | 30 / 30 / 30 | 30 / 30 / 30 | 0.83 [0, 1.0, 2] |
| V4 TEST | 180 | 8.57 [3, 8.5, 15] | 1.805 | 144 / 36 | 60 / 60 / 60 | 60 / 60 / 60 | 0.86 [0, 1.0, 2] |
