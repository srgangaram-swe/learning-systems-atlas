# Reinforcement-learning comparison

Training-validation selects policies; untouched test never controls selection.

| Method | Validation return | Test return | Training-seed SD | Seeds >=475 |
|---|---:|---:|---:|---:|
| reinforce | 500.00 | 498.73 | 1.41 | 3 |
| dqn | 498.83 | 498.19 | 1.78 | 3 |
| dqn_recent | 339.77 | 351.14 | 120.55 | 1 |
| dqn_online_target | 137.60 | 130.19 | 209.43 | 0 |
| dqn_recent_online | 9.10 | 9.29 | 0.04 | 0 |
| ppo | 500.00 | 478.03 | 38.05 | 2 |
| random | 0.00 | 23.25 | 0.67 | 0 |

The 475 threshold applies only to the full 500-step CartPole profile.
Seed-level bootstrap intervals are descriptive and fragile with few policies.
See evidence.json for every seed and measured interaction cost.
