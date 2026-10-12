### Game size

| public nodes | decision | chance | fold | showdown | fiber (hands) | info sets | levels |
|---|---|---|---|---|---|---|---|
| 54957 | 21380 | 339 | 15258 | 17980 | 10 | 171920 | 13 |

### E1 exact solve (exploitability in mA/hand = milli-antes per hand; % of the 2-ante initial pot)

| iter | CFR+ s | CFR+ expl mA/h | CFR+ % pot | CFR s | CFR expl mA/h | CFR % pot |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 2329.49 | 116.474 | 0.1 | 2329.49 | 116.474 |
| 2 | 0.2 | 2351.12 | 117.556 | 0.1 | 2286.67 | 114.333 |
| 5 | 0.4 | 1653.58 | 82.679 | 0.3 | 1879.34 | 93.967 |
| 10 | 0.7 | 808.67 | 40.433 | 0.6 | 1168.61 | 58.430 |
| 20 | 1.4 | 400.97 | 20.049 | 1.3 | 683.92 | 34.196 |
| 50 | 3.3 | 135.37 | 6.769 | 3.3 | 371.85 | 18.592 |
| 100 | 6.4 | 56.31 | 2.815 | 6.6 | 222.91 | 11.146 |
| 200 | 13.1 | 20.41 | 1.021 | 13.1 | 127.20 | 6.360 |
| 300 | 19.6 | 10.52 | 0.526 | 19.7 | 90.42 | 4.521 |
| 500 | 32.8 | 4.64 | 0.232 | 33.5 | 58.40 | 2.920 |
| 1000 | 67.1 | 1.42 | 0.071 | 66.9 | 31.62 | 1.581 |
| 1500 | 99.8 | 0.69 | 0.034 | 101.0 | 22.36 | 1.118 |
| 2000 | 134.0 | 0.41 | 0.020 | 135.6 | 17.36 | 0.868 |

seconds/iteration: CFR+ 0.0670, vanilla 0.0678; solver arrays 24.7 MB; peak RSS 198 MB; game value for P0 -0.11411 antes/hand.

### E2a turn-boundary value function (errors in units of x = each player's pot share)

turn oracle labels: 22000 situations, 11.2 CPU-ms each, mean label exploitability 0.00063 x; exact flop labels: 255 CPU-ms each. On-policy states: 2211.

| V_turn | N | residual | range-w loss | train s | test RMSE | test RMSE range-w | on-policy RMSE | on-policy RMSE range-w |
|---|---|---|---|---|---|---|---|---|
| 200 | 200 | True | False | 10.0 | 1.004 | 0.466 | 0.471 | 0.368 |
| 2000 | 2000 | True | False | 14.3 | 0.824 | 0.348 | 0.367 | 0.291 |
| 20000 | 20000 | True | False | 14.2 | 0.581 | 0.230 | 0.267 | 0.227 |
| 20000_raw | 20000 | False | False | 13.3 | 0.661 | 0.315 | 0.334 | 0.303 |
| 20000_rw | 20000 | True | True | 14.2 | 0.656 | 0.151 | 0.214 | 0.159 |
| checkdown | 0 | - | - | - | 0.948 | 0.274 | 0.381 | 0.313 |

### E2b flop-boundary value function (bootstrapped labels; errors vs EXACT flop labels)

| V_flop | M_flop | boot label CPU-ms | boot-label RMSE vs exact | range-w | test RMSE | test RMSE range-w | on-policy RMSE | on-policy range-w |
|---|---|---|---|---|---|---|---|---|
| 200 | 200 | 87.5 | 0.414 | 0.206 | 1.062 | 0.569 | 0.467 | 0.403 |
| 2000 | 2000 | 100.6 | 0.320 | 0.139 | 0.792 | 0.393 | 0.300 | 0.277 |
| 20000 | 2000 | 122.5 | 0.233 | 0.083 | 0.794 | 0.360 | 0.309 | 0.292 |
| 20000_raw | 2000 | 88.1 | 0.282 | 0.145 | 0.891 | 0.441 | 0.379 | 0.356 |
| 20000_rw | 2000 | 115.9 | 0.241 | 0.072 | 0.838 | 0.227 | 0.272 | 0.225 |
| checkdown | - | - | - | - | 1.050 | 0.363 | 0.491 | 0.460 |

### E2c continual re-solving agents (K = 300 CFR+ iterations per re-solve, both seats)

| agent | expl mA/h | % pot | seat0 expl | seat1 expl | ms per re-solve (batched) pre/flop/turn | extraction s | # re-solves |
|---|---|---|---|---|---|---|---|
| full solve, 300 it | 10.5 | 0.53 | - | - | - | - | - |
| full solve, 2000 it | 0.41 | 0.020 | - | - | - | - | - |
| oracle_naive | 520.5 | 26.03 | 757.3 | 283.7 | 3312.3 / 74.8 / 3.2 | 154.8 | 16237 |
| oracle_safe | 23.7 | 1.18 | 24.7 | 22.7 | 3252.5 / 76.8 / 3.4 | 158.3 | 16237 |
| checkdown_safe | 532.5 | 26.63 | 393.8 | 671.2 | 52.5 / 9.0 / 3.4 | 59.5 | 16237 |
| learned200_safe | 521.2 | 26.06 | 549.9 | 492.5 | 87.2 / 33.0 / 3.3 | 74.9 | 16237 |
| learned2000_safe | 511.8 | 25.59 | 441.7 | 582.0 | 71.9 / 24.9 / 2.8 | 60.9 | 16237 |
| learned20000_safe | 499.1 | 24.95 | 469.5 | 528.7 | 71.0 / 23.9 / 2.8 | 60.3 | 16237 |
| learned20000_raw_safe | 506.0 | 25.30 | 488.6 | 523.4 | 63.6 / 18.1 / 2.7 | 54.3 | 16237 |
| learned20000_rw_safe | 417.4 | 20.87 | 412.6 | 422.2 | 75.1 / 23.6 / 2.6 | 57.7 | 16237 |
| learned20000_naive | 824.4 | 41.22 | 1057.8 | 590.9 | 72.1 / 23.3 / 3.1 | 64.7 | 16237 |

### E2d exact leaf at one boundary, approximate leaf at the other (safe re-solving)

| agent | expl mA/h | % pot | extraction s (both seats) |
|---|---|---|---|
| checkdown@preflop + oracle@flop | 416.5 | 20.82 | 101.2 |
| oracle@preflop + checkdown@flop | 379.9 | 18.99 | 106.3 |
| learned20000_rw@preflop + oracle@flop | 215.6 | 10.78 | 106.1 |
| oracle@preflop + learned20000_rw@flop | 258.7 | 12.94 | 102.5 |

### E3 off-tree opening bet by P0 at every flop root (we are P1); loss = P0 best-response value - value (mA/hand)

Blueprint in the base game: P0 best response gains 0.53 mA/h over the game value.

| size (pot) | value gain of size for P0 | aug-solve expl | snap down | snap up | snap nearest | safe re-solve | aug Nash | re-solve ms/root (batched) | single re-solve s | reached P1 nodes w/o translation (down/up/near) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 7.12 | 1.56 | 389.1 | 389.1 | 389.1 | 18.8 | 1.8 | 318.0 | 0.66 | 360/360/360 of 3700 |
| 0.75 | -0.07 | 1.60 | 227.2 | 179.0 | 227.2 | 7.2 | 2.2 | 277.7 | 0.67 | 0/0/0 of 1860 |
| 1.5 | 0.01 | 1.50 | 50.8 | 235.5 | 50.8 | 1.4 | 1.9 | 256.4 | 0.60 | 0/720/0 of 750 |

### E4 robustness vs exploitation (seat-averaged, mA/hand; RNR with K = 600 CFR+ iterations)

| model | strategy | EV vs model | exploitability | EV as P0 | EV as P1 |
|---|---|---|---|---|---|
| station | Nash (p=0) | 811.9 | 0.4 | 694.9 | 929.0 |
| station | RNR p=0.25 | 2861.8 | 154.4 | 2922.9 | 2800.6 |
| station | RNR p=0.5 | 3085.9 | 278.3 | 3078.2 | 3093.5 |
| station | RNR p=0.75 | 3122.8 | 341.9 | 3120.3 | 3125.4 |
| station | RNR p=1.0 | 3142.8 | 2526.3 | 3142.8 | 3142.8 |
| station | exact best response | 3142.9 | 1570.2 | 3142.9 | 3142.9 |
| maniac | Nash (p=0) | 540.9 | 0.4 | 164.9 | 916.9 |
| maniac | RNR p=0.25 | 1178.0 | 47.1 | 1127.3 | 1228.6 |
| maniac | RNR p=0.5 | 1298.0 | 137.9 | 1306.2 | 1289.7 |
| maniac | RNR p=0.75 | 1428.6 | 334.4 | 1428.6 | 1428.6 |
| maniac | RNR p=1.0 | 1428.6 | 2302.2 | 1428.6 | 1428.6 |
| maniac | exact best response | 1428.6 | 1498.4 | 1428.6 | 1428.6 |
| nit | Nash (p=0) | 421.6 | 0.4 | 316.6 | 526.5 |
| nit | RNR p=0.25 | 1393.2 | 108.1 | 1016.2 | 1770.3 |
| nit | RNR p=0.5 | 2900.5 | 1114.5 | 2971.2 | 2829.7 |
| nit | RNR p=0.75 | 3296.2 | 1591.4 | 3234.7 | 3357.7 |
| nit | RNR p=1.0 | 3344.4 | 4920.2 | 3284.1 | 3404.7 |
| nit | exact best response | 3344.4 | 9039.3 | 3284.1 | 3404.8 |

### E5 timing

| quantity | value |
|---|---|
| full solve: s/iteration | 0.0670 |
| full solve to <1% pot (iter, s) | [300, 19.581927061080933] |
| full solve to <0.1% pot (iter, s) | [1000, 67.09361338615417] |
| turn oracle label, CPU-ms / situation | 11.2 |
| exact flop label, CPU-ms / situation | 255.0 |
| bootstrapped flop label (200), CPU-ms / situation | 87.5 |
| bootstrapped flop label (2000), CPU-ms / situation | 100.6 |
| bootstrapped flop label (20000), CPU-ms / situation | 122.5 |
| bootstrapped flop label (20000_raw), CPU-ms / situation | 88.1 |
| bootstrapped flop label (20000_rw), CPU-ms / situation | 115.9 |
| train V_turn 200, s | 10.0 |
| train V_turn 2000, s | 14.3 |
| train V_turn 20000, s | 14.2 |
| train V_turn 20000_raw, s | 13.3 |
| train V_turn 20000_rw, s | 14.2 |
| train V_flop 200, s | 10.6 |
| train V_flop 2000, s | 12.6 |
| train V_flop 20000, s | 12.9 |
| train V_flop 20000_raw, s | 12.9 |
| train V_flop 20000_rw, s | 12.9 |
| single re-solve preflop root, checkdown leaf, K=300 (57 nodes), s | 0.162 |
| single re-solve preflop root, learned leaf, K=300 (57 nodes), s | 0.376 |
| single re-solve preflop root, oracle leaf, K=300 (54957 nodes), s | 16.627 |
| single re-solve flop root, checkdown leaf, K=300 (57 nodes), s | 0.154 |
| single re-solve flop root, learned leaf, K=300 (57 nodes), s | 0.325 |
| single re-solve flop root, oracle leaf, K=300 (2082 nodes), s | 0.593 |
| single re-solve turn root, checkdown leaf, K=300 (57 nodes), s | 0.105 |
| single re-solve turn root, learned leaf, K=300 (57 nodes), s | 0.104 |
| single re-solve turn root, oracle leaf, K=300 (57 nodes), s | 0.103 |
| E3 single flop re-solve with off-tree size 0.25, s | 0.663 |
| E3 single flop re-solve with off-tree size 0.75, s | 0.673 |
| E3 single flop re-solve with off-tree size 1.5, s | 0.599 |

### Label noise: per-hand CFVs vs a 1000-iteration solve (1000 random turn situations, x units)

| CFR+ iters | label expl (x) | RMSE all hands | RMSE hands with p>1e-3 | range-weighted RMSE |
|---|---|---|---|---|
| 50 | 0.00828 | 0.267 | 0.249 | 0.074 |
| 100 | 0.00293 | 0.197 | 0.174 | 0.043 |
| 200 | 0.00105 | 0.131 | 0.112 | 0.023 |
| 400 | 0.00039 | 0.066 | 0.054 | 0.010 |

