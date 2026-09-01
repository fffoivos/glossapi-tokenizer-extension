# Clariden scheduler-capacity probe — 2026-08-02

Probe time: `2026-08-02T11:19:46+02:00`

This is a scheduler estimate, not a reservation and not a measured training
throughput result.

Observed `normal` partition policy:

- `MaxNodes=UNLIMITED`
- `MaxTime=12:00:00`
- `OverSubscribe=EXCLUSIVE`
- allowed QoS: `normal,stop`
- account/user: `a0140` / `fffoivos`

The node-state snapshot at the receipt time included 362 idle nodes. Node state
is volatile and must be refreshed before launch.

Non-submitting `sbatch --test-only` requests at the full 12-hour limit predicted
starts at approximately `12:23:46`–`12:23:48`, about 64 minutes after the
probe, for all of the following aggregate allocations:

| Requested nodes | Intended geometry |
|---:|---|
| 20 | five DP=16 arms |
| 40 | five DP=32 arms |
| 80 | five DP=64 arms |
| 160 | five DP=128 arms |

Earlier probes in the same session also predicted near-term starts for 5 and
10 nodes. No job was submitted by these probes.

Command shape:

```bash
sbatch --test-only \
  --account=a0140 \
  --partition=normal \
  --nodes=<N> \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --time=12:00:00 \
  --wrap='true'
```

Interpretation: the project is not policy-limited to the earlier one-to-three
node plan, and the scheduler forecast near-term capacity for the
20/40/80/160-node candidate campaign shapes at this instant. Only the actual
DP=16/32/64/128 training benchmark can establish which size is fastest and
whether the five-arm round fits the target.
