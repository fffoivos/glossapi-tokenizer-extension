# Full G1 and deterministic G2 execution record — 2026-07-19

This record freezes the development-only outcome of the full G1 decoder sweep
and the deterministic G2 heading-window sweep. No sealed labels, consensus
receipt, or `40_frozen` artifact was used for candidate construction,
selection, or audit.

## Root and code

Clariden root:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/bibliography_evolution_launch_20260718
```

- Full-G1 packet code: `c5b79d98bf58cf9111d49103d889d990be9018c6`.
- Full-G1-bound G2 code: `ad0fd422658327034a2aaf10b4e9fa77ea25e825`.
- G2 no-admission finalizer: `7df2344ac123d290115a496025232661d0854a6d`.

## Full G1

The already materialized 27-row packet was independently audited before
submission:

```text
packets/g1-full-c5b79d98-prep-2790166/
queue.g1.full.jsonl
```

- Queue SHA-256:
  `b56cb01317dd8408aaefef4a1ce6f1fe1cd1c221281a09f46169a855597cafca`.
- Slurm array: `2793581`; tasks 0--26 all `COMPLETED 0:0`.
- The 27 receipts and all 459 owned files passed independent immutable-file,
  recursive-lineage, leakage, test, and artifact verification.
- No child weakly dominated the 0.30 control on token FP, token FN, spurious
  blocks per zero-block document, and mean emitted-line boundary error.

The audited 33-candidate full-G1 registry is:

```text
registries/g0-g1-full-audited-2793581-da7dfcf/development_registry.json
```

- Registry SHA-256:
  `eb6f229e8ce0b8df68c06461d580a9db807840a1f546b7955c961cfd0ec7285b`.
- 33 candidates, 33 eligible, 12 Pareto.
- Finalizer job `2793808` retained
  `g1-1909806a497053bb7ac4c964` as the G2 sequencing parent.
- Parent receipt SHA-256:
  `9ae3ce4f3d80676ef7d561e429c835e12c65690f9004da15e0dc4e0a0e4479fb`.
- Parent prediction SHA-256:
  `58c4f0a4108f1c7c461782c81274363bb29e83fea2b9151dcd7751aecd6da684`.
- Parent barrier SHA-256:
  `540ad1326cc282aed13f4c13458d002bc13288fba3d765cbb78ff48fbb2c09b1`.

## Deterministic G2

Prep job `2793822` regenerated the text-only heading roles and materialized a
fresh packet bound to the full-G1 parent:

```text
packets/g2-deterministic-ad0fd422-prep-2793822/
queue.g2.deterministic.jsonl
```

- Packet receipt SHA-256:
  `24fdb60e25263abd747a93fbce20c8b8e70df5a89d851034150e2b798e7099ec`.
- Queue SHA-256:
  `253ca9edc87958490cc302b38774347b9a964ff8f545f59ea777c0c96a06ba63`.
- Role counts: 144 `BIB_HEADER`, 55 `BIB_SUBHEADER`, 196
  `NON_BIB_HEADER`.
- Role-vector SHA-256:
  `ba2b8dd106383087aa91ae51bccc57b6a0bc9a1463730588f3e9716f6bb9d607`.
- The 41-test attestation, packet bindings, inputs, and four queue rows passed
  independent pre-submit audit.

Clariden CPU array `2793872` ran windows 1--4 in parallel. All four tasks
completed `0:0` in 1m28s and passed independent post-run verification.

| window | candidate | receipt SHA-256 | FP delta | FN delta | boundary delta |
|---:|---|---|---:|---:|---:|
| 1 | `g2-e0da48d37e3d357b45603366` | `4a10639f49dcdd957a2ea20bd3ea5acc7f6c02bb4afad6ce5669cfa6d1b447ed` | -4,020 | +113,212 | -0.166693 |
| 2 | `g2-f555b4bafa3befb2c6ed94e9` | `ef734cae8dc03c8dfd65c165174b1d2a87bcf21361fdc7efa0275ff93e1dab80` | -4,316 | +113,208 | -0.199373 |
| 3 | `g2-aa052a5aa2898cf11a7596e2` | `f809837611ee51f7a410f5ac339fbd0701b5b27a4881a0fb24d2a2d6a30a42ea` | -4,316 | +113,208 | -0.199373 |
| 4 | `g2-7c00e09c62c7745aa5906af4` | `96a3de93ad05916a723013c462b228952e3cce13773e032179e97de5b56eed9d` | -4,316 | +113,208 | -0.199373 |

The reference parent objective vector is token FP `101547`, token FN `39053`,
spurious blocks `0.037037037037037035`, and boundary error
`2.570287539936102`. Every G2 candidate improves FP and boundary error but
materially worsens FN. Windows 2--4 produce identical predictions.

## Frozen outcome

Registry build job `2793902` wrote a 37-candidate, 37-eligible registry with 12
Pareto candidates; none is G2. The registry SHA-256 is
`c629daa33b88ed32a7afd2353935ba90c2efb03de39856a59d83a13fe5e444e9`.

The finalizer passed three focused tests in Clariden job `2793908`. Job
`2793911` then recursively verified all 37 receipts and wrote:

```text
registries/g0-g2-audited-2793872-7df2344/
  development_registry.json
  g2_selection.json
  receipt.json
```

- Selection SHA-256:
  `75ff6ec4dd77274e6943856e77f099c83cae41416bf7961d854766e6208144c2`.
- Receipt SHA-256:
  `30db0bd97e1f06576a637308b90ca5fa9005bd1ac50ba60d68955bda2b0c08c9`.
- Promoted G2 candidate: none.
- Retained parent: `g1-1909806a497053bb7ac4c964`.
- `g3_authorized: false`.

This is a valid negative result. The deterministic heading-window component
must not advance under the frozen strict-component-isolation rule. A different
component or architecture requires a new predeclared generation and fresh
development evidence; it must not be selected using the sealed set.
