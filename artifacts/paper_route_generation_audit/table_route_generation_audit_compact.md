# Table A (compact). V4 TEST route generation

Compact Table A. The first block counts source instances (2090 = 180 + 1910). The second block counts candidate-route evaluations, not source instances. The four reason rows sum to 6322, and the last two rows are a separate partition of the same events (157 + 6165). Unverified timeout and unverified search exhausted mean the certificate search did not return a witness. They are not proofs of infeasibility. The full audit, including accepted layout, source size, length bin, and charge class, remains in `table_route_generation_audit`.

| Counting unit | Item | Count | Note |
|---|---|---:|---|
| Source instance | Candidate instances examined | 2090 | Sum of the next two rows |
| Source instance | Instances with one accepted route | 180 | At most one route is kept per instance |
| Source instance | Instances with no accepted route | 1910 |  |
| Route evaluation | Rejection events | 6322 | Candidate vehicle routes, not source instances |
| Route evaluation | Outside length bins | 1726 | Frozen-route length outside 3–15 customers |
| Route evaluation | Quota full | 3674 | Layout, length, and charge-class cell already filled |
| Route evaluation | Unverified timeout | 822 | Search ended without a witness; not infeasibility |
| Route evaluation | Unverified search exhausted | 100 | Search ended without a witness; not infeasibility |
| Route evaluation | Events on instances later accepted | 157 | Earlier vehicle routes of an accepted instance |
| Route evaluation | Events on instances with no accepted route | 6165 |  |
