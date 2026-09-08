# Finite discounted planning: specification and convergence argument

## Contract

Let S and A be finite, P(s'|s,a) nonnegative with each row summing to one,
and rewards bounded. Let 0 <= gamma < 1. Terminal states have value zero;
their outgoing rewards are not accumulated after termination. The implementation
bounds S <=256, A <=16 and gamma <=0.9999. Computation is floating point, not
exact real arithmetic.

For nonterminal s define

`(TV)(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma 1_nonterminal(s') V(s')]`.

For terminal s, `(TV)(s)=0`. The program returns values V, a deterministic
first-argmax policy, and a **recomputed** residual `||TV-V||_infinity <= epsilon`.
An iteration cap reached without that certificate raises `ReinforcementError`.

## Contraction and value iteration

For any U,V, the difference between two maxima is bounded by the maximum
absolute difference of their arguments. Nonnegative probabilities and total
mass at most one over nonterminal next states therefore give

`||TU-TV||_infinity <= gamma ||U-V||_infinity`.

The finite-dimensional real vector space with the sup norm is complete. The
contraction theorem gives a unique fixed point V*, and repeated Bellman updates
converge to it from any finite initialization. After k updates the error is at
most `gamma^k ||V0-V*||`. This is a mathematical convergence statement under
the assumptions, not a claim that a finite collection of tests proves it.

The triangle inequality and contraction also give

`||V-V*|| <= ||V-TV|| + gamma ||V-V*||`,

hence the returned residual certifies `||V-V*|| <= epsilon/(1-gamma)` in exact
arithmetic. Floating-point rounding adds numerical error; tests compare both
independent planners with tolerances and inspect the residual, not golden bits.

## Policy evaluation and improvement

For a fixed deterministic policy pi, remove terminal continuation from P_pi and
set terminal immediate rewards to zero. The spectral radius of gamma P_pi is
at most gamma <1. Thus `I-gamma P_pi` is nonsingular. Solve the linear system
`(I-gamma P_pi)V_pi = r_pi` directly; do not form a matrix inverse.

The greedy improvement pi' satisfies `T_pi' V_pi >= V_pi`. Monotonicity and
contraction imply `V_pi' >= V_pi`. If the optimal Bellman residual is nonzero,
at least one state improves strictly under exact evaluation. There are finitely
many deterministic policies; fixed tie-breaking and strict improvement before
optimality prevent a cycle of distinct improving value functions. At zero
residual the value is the unique V*. The numerical implementation terminates
on the residual, not apparent policy equality; bounded failure remains explicit.

## Cost, tests, and limits

Dense value iteration costs O(K S² A) time and O(S² A) model storage. Policy
iteration adds O(S³) linear solves per improvement. This is intentionally not
a large sparse-MDP or continuous-control solver.

`test_foundations.py` includes a hand-derived corridor, independent value/policy
iteration agreement over generated slip probabilities, a contraction property,
Gymnasium checker and transition/reward consistency, malformed MDPs, and injected
linear-solve failure. Unreachable goals and cliffs do not break discounted
existence; they may produce poor optimal values and are not silently repaired.
