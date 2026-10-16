# Table B. Frozen-route characteristics

TRAIN and VALIDATION are the SynthCharge development corpora. TEST is the fresh V4 corpus only. Customer counts describe the frozen customer sequence. Mean customer-only distance is the mean length of the frozen depot–customers–depot sequence in normalized SynthCharge coordinate units. Instance coordinates lie in the generator's unit square. These distances are not kilometers. Station-visit statistics describe the stored charging certificate, not FA-HPPO decisions. A no-charge-required label means a continue-only replay completed. In these corpora that label coincides with zero certificate station visits, but the certificate remains one feasible witness rather than an FA-HPPO trajectory.

| Split | Routes | Customers mean [min, median, max] | Mean customer-only distance (coordinate units) | Charging required | No charge required | C / R / RC | Short / medium / long | Certificate station visits mean [min, median, max] |
|---|---:|---|---:|---:|---:|---|---|---|
| TRAIN | 180 | 8.68 [3, 8.5, 15] | 1.744 | 135 | 45 | 60 / 60 / 60 | 60 / 60 / 60 | 0.77 [0, 1.0, 3] |
| VALIDATION | 90 | 8.50 [4, 8.5, 15] | 1.820 | 72 | 18 | 30 / 30 / 30 | 30 / 30 / 30 | 0.83 [0, 1.0, 2] |
| V4 TEST | 180 | 8.57 [3, 8.5, 15] | 1.805 | 144 | 36 | 60 / 60 / 60 | 60 / 60 / 60 | 0.86 [0, 1.0, 2] |
