# V2 route-construction diagnostic

Consumed V1 TEST parents are listed in the CSV with V2 counts withheld.
This is not an optimality claim and does not force V2 counts to match official plans.

- official instance variants: 76
- development variants compared: 64
- official tours on development variants: 208
- original PyVRP routes on development variants: 140
- V2 certified routes on development variants: 311

Admitted benchmark routes are complete certified partitions only.
An unresolved source route is quarantined with all of its customers.

- original development routes: 209
- admitted certified routes: 423
- search timeouts: 373
- search exhaustions: 432
- customer coverage: {"coverage_fraction": 0.9571428571428572, "customers_in_admitted_partitions": 1273, "quarantined_customers": 57, "quarantined_source_routes": 9, "silently_dropped_customers": 0, "total_source_customers": 1330}
