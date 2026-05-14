# Greek-Share Path-A Runbook

Goal: produce exact Greek-token counts for every Apertus-pretraining
Greek-bearing dataset, then derive Greek's share of the Apertus-8B-2509
realised pretraining budget (~13.4 T).

Anchors:
- Plan + denominator + filter recipe: `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`.
- Scripts: this folder (`tokenize_greek_slice.py`, `aggregate.py`, `entrypoint.sh`).

## 1. Instance bring-up

Zone: `europe-west4-b`. Same as the existing apertus-greek-tokenizer
m3-megamem-64 (which is unrelated to this run and must not be touched).

```bash
ZONE=europe-west4-b
NAME=apertus-greek-share-$(date -u +%Y%m%dt%H%M%Sz)
PROJECT=$(gcloud config get-value project)

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=c4-highcpu-192 \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200 \
  --boot-disk-type=pd-balanced \
  --local-ssd=interface=NVME \
  --local-ssd=interface=NVME \
  --local-ssd=interface=NVME \
  --local-ssd=interface=NVME \
  --labels=owner=foivos,workload=greek-share-tokenization,run-date=$(date -u +%Y%m%d) \
  --metadata=enable-oslogin=TRUE

# c4 supports 4 local-SSD slices x 375 GB = 1.5 TB scratch as a stripe.
```

## 2. Push scripts + env

```bash
gcloud compute scp --zone="$ZONE" --recurse \
  /home/foivos/Projects/glossapi-tokenizer-extension/ops/greek_share_run/ \
  "$NAME":~/greek_share_run/

gcloud compute ssh --zone="$ZONE" "$NAME" --command "\
  export HF_TOKEN='$HF_TOKEN' && \
  bash ~/greek_share_run/entrypoint.sh"
```

The entrypoint:
- installs apt deps + Python deps,
- formats and mounts the first local SSD at `/mnt/data` (subsequent SSDs left raw — the FineWeb2-HQ Greek slice is 83 GB so we don't need to stripe all 4),
- writes `/mnt/data/profile.sh` with `HF_TOKEN`, `RAYON_NUM_THREADS=192`, `TOKENIZERS_PARALLELISM=true`, HF cache redirects,
- pre-downloads the Apertus tokenizer.

## 3. Run tokenization (one driver per dataset)

Open a long-lived tmux session on the worker:

```bash
gcloud compute ssh --zone="$ZONE" "$NAME" --command "tmux new -d -s greek 'bash'"
gcloud compute ssh --zone="$ZONE" "$NAME"
# inside ssh:
tmux a -t greek
source /mnt/data/profile.sh
cd ~/greek_share_run

# datasets in increasing size — quick sanity runs first
python3 tokenize_greek_slice.py euroblocks_el          2>&1 | tee -a /mnt/data/logs/euroblocks_el.log
python3 tokenize_greek_slice.py clean_wikipedia_el     2>&1 | tee -a /mnt/data/logs/clean_wikipedia_el.log
python3 tokenize_greek_slice.py europarl_el            2>&1 | tee -a /mnt/data/logs/europarl_el.log
python3 tokenize_greek_slice.py paradocs_el            2>&1 | tee -a /mnt/data/logs/paradocs_el.log
python3 tokenize_greek_slice.py institutional_books_el 2>&1 | tee -a /mnt/data/logs/institutional_books_el.log
python3 tokenize_greek_slice.py fineweb2_hq_ell        2>&1 | tee -a /mnt/data/logs/fineweb2_hq_ell.log
```

Expected wall times (CPU-bound on c4-highcpu-192):
- `euroblocks_el`        — under 1 min
- `clean_wikipedia_el`   — 1–2 min
- `europarl_el`          — 10–20 min (20 Greek pairs)
- `paradocs_el`          — 30–60 min (large)
- `institutional_books_el` — 30–60 min (long-document OCR text)
- `fineweb2_hq_ell`      — 30–60 min (83 GB parquet, dominant)

## 4. Aggregate

```bash
python3 aggregate.py
cat /mnt/data/outputs/summary.json
```

## 5. Pull results to `home`

```bash
gcloud compute scp --zone="$ZONE" --recurse \
  "$NAME":/mnt/data/outputs/ \
  /home/foivos/Projects/glossapi-tokenizer-extension/ops/greek_share_run/outputs/
```

Commit the `outputs/*.json` + `summary.json` so the number is reproducible
from the repo state.

## 6. Tear-down (NON-OPTIONAL)

```bash
gcloud compute instances delete --zone="$ZONE" "$NAME" --quiet
```

Spot c4-highcpu-192 in europe-west4 ≈ $4–5/hr. End-to-end run budget at
2–3 hr ≈ $8–15. Idle leakage is the worst-case (per project methodology
Rule 9) — delete the instance the moment outputs land on `home`.

## 7. Sanity checks before publishing the number

- Apertus tokenizer vocab size = 131,072 (Apertus base) and matches what
  `entrypoint.sh` reports.
- Each per-dataset `*.json` has non-zero `docs` and `tokens_with_bod_eod`.
- `summary.json.overall.greek_share_pct` is plausible (expect 0.1%–1.5%
  range).
- Spot-check 100 random Greek docs from FineWeb2-HQ for visual confirmation
  they are actually Greek (not script-only / not encoded).
