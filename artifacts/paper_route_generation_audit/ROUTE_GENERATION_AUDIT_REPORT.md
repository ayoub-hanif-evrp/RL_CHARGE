# Fixed-route construction audit

This note documents the existing route-generation process. No routes were regenerated, no policy was trained, and the consumed V4 TEST was not re-evaluated.

## Authoritative inputs

- TRAIN and VALIDATION: `data/routes_v2/synthcharge_final/{train,validation}/`, built from `configs/v2/synthcharge_final_benchmark.json` (seed starts 100000 and 200000).
- V4 TEST: `data/routes_v2/synthcharge_v4_test/`, built from `results/v4/test/configs/synthcharge_v4_test.json` (seed start 500000) by `scripts/v4/test/generate_v4_test.py`.
- The V2 configuration also declares a test seed start of 300000. That range is not the V4 TEST set and was not analyzed here. The V3 TEST corpus was not opened.
- PyVRP builds customer-only pickup-VRPTW routes (`src/routing/pyvrp_generator.py`). The charging certificate is a best-first search for one simulator-valid trajectory (`src/exact/charging_certificate.py`); its `exact` flag is false.

## Figure A

Selection rule: among VALIDATION routes labeled charging-required, with 6 to 10 customers, and with both the source instance file and a certificate trace available, choose the lowest generator seed and then the route identifier.

Selected route: `v2sc_sc_R_n15_s200000_00` (seed 200000, layout R, 8 customers, source instance `sc_R_n15_s200000`).

Fixed-route construction on one existing SynthCharge validation instance (v2sc_sc_R_n15_s200000_00; generator seed 200000; 8 customers on the frozen route). (a) PyVRP solves a customer-only capacitated pickup vehicle-routing problem with time windows. The solid arrows follow the selected customer sequence from the depot back to the depot. Charging stations are available facilities and are not inserted into this route. Faint markers are source-instance customers assigned to other vehicle routes and therefore not part of the frozen sequence. (b) The same customer order with the stored charging-certificate trajectory. Dashed arrows are charging detours (2 station visit(s) in this witness); station labels report the target state of charge recorded in that trace. This panel is an existing feasible charging witness from the certificate search. It is not a globally optimal charging plan and it was not produced by FA-HPPO. The customer visiting order is unchanged between panels.

## Table A counting units

- Candidate instances examined: 2090.
- Instances with one accepted route: 180.
- Instances with no accepted route: 1910.
- Rejection events: 6322. These are candidate vehicle routes rejected inside an instance, including routes examined before a later route from the same instance was accepted (157 such events).
- `unverified_timeout` and `unverified_search_exhausted` are incomplete searches, not infeasibility certificates.
- Acceptance is balanced on layout, frozen-route length (short 3-5, medium 6-10, long 11-15), and charging requirement. Source-instance sizes 15, 30, and 50 are generation settings, not the balancing factors.

## Table B

Customer-only distance is the stored `route_distance` of the depot-customer-depot sequence. It matched `total_distance` on every analyzed route.
Certificate station visits count `CHARGE` steps in the stored trace. In these three corpora, every no-charge-required route has zero certificate station visits, and every charging-required route has at least one. The label is still not defined by that count: no-charge-required means a continue-only replay completed. The stored trace is one feasible witness, not the continue-only trajectory and not an FA-HPPO decision.

## Figure B

The pipeline figure separates instance generation, customer-only PyVRP routing, certificate search, quota filtering, freezing, and the later FA-HPPO charging controller.

## Checks against metadata and quotas

- No disagreement was found among corpus counts, metadata, audit IDs, length bins, and the declared quotas.

## Provenance

- Git commit: `14a13b6ebcef3b2f06811713d0df5f5cb08abd34`
- Input SHA-256 values are listed in `provenance.json`.

## Limitations

- The illustration is one validation route chosen by a disclosed rule. It is not a performance example and is not a TEST evaluation.
- Non-selected customers are the source-instance customers omitted from the accepted vehicle route. Their assignment to other PyVRP routes is not redrawn.
- Certificate target-SOC annotations are the values stored in the witness, not decisions of FA-HPPO.
- Rejection counts must not be read as a failure rate over source instances.
