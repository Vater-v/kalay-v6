"""What step / step_target do with an illegal action (legality is documented as the caller's precondition):
a masked grid size through step, an over-stack amount through step_target, and whether guards.target rejects it.
Usage: illegal_step.py OUT.json"""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
from poker.envs import config, guards
from poker.envs.betting import init
from poker.envs.actions import action_table
from poker.envs.transition import step, step_target
from poker.envs.evaluator import get_rewards
from jax.experimental import checkify
out = {}
s = init(jax.random.PRNGKey(0), 2, jnp.asarray([300, 300], jnp.int32))   # 3 BB stacks
m, t = action_table(s)
big = config.ALLIN - 1                                                     # 5/2 pot: masked (>= stack)? check
out["mask"] = np.asarray(m).tolist(); out["targets"] = np.asarray(t).tolist()
illegal = [a for a in range(2, config.ALLIN) if not m[a]]
a = illegal[-1] if illegal else big
s2 = step(s, jnp.int32(a))
out["masked_grid_action"] = a
out["after_step"] = {"stack": np.asarray(s2.stack).tolist(), "bet_street": np.asarray(s2.bet_street).tolist(),
                     "chips_total": int(np.asarray(s2.stack).sum() + np.asarray(s2.bet_total).sum()), "chips_before": 600}
err, _ = jax.jit(checkify.checkify(step))(s, jnp.int32(a))
out["checkify_step_error"] = str(err.get())
s3 = step_target(s, jnp.int32(2), jnp.int32(5000))
out["step_target_over_stack"] = {"stack": np.asarray(s3.stack).tolist(), "bet_street": np.asarray(s3.bet_street).tolist()}
try:
    guards.target(s, 2, 5000); out["guards_target_over_stack"] = "accepted"
except ValueError as e:
    out["guards_target_over_stack"] = "rejected: " + str(e)[:120]
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
