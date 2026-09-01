# DP32 restart acceptance disclosure

The restart-gradient threshold in `configs/execution_profiles.json` was not
predeclared. Commit `71d1bba2` added the `0.001` absolute and `0.02` relative
tolerances after the DP32 restart result existed and before the promotion
receipt was frozen. It must therefore never be cited as an independently
predeclared acceptance criterion or as proof of bitwise equality.

The frozen receipt records, at update 161:

- uninterrupted DP32 loss `2.204937` and restarted loss `2.204937`;
- uninterrupted parameter norm `7142.029` and restarted parameter norm
  `7142.029`;
- uninterrupted gradient norm `0.873` and restarted gradient norm `0.881`, an
  absolute delta of `0.008` (about `0.916%` of the uninterrupted value);
- exact DP64 equality for all three logged fields.

The earlier explanation attributing the DP32 difference to collective
reduction order is unsupported by this receipt and is withdrawn. The cause of
the logged gradient-norm difference is not established.

This does not change the parallelism choice: DP64 was rejected independently
because its loss trajectory failed both drift gates, and production uses the
DP32 control profile. The run's recovery evidence is described conservatively
as numerically continuous, not bitwise-exact, based on exact logged loss and
parameter norm, the finite bounded gradient difference, and the separate
hardware graceful-stop/resume smoke. Every production segment boundary must
still pass its checkpoint and loss-continuity audit.
