#!/bin/bash
# Push the 15_clean_academic code to Clariden scratch and build the aarch64 Rust binary.
# Run from a host that can reach CSCS (home when connectivity is up, else the Mac relay).
# home→CSCS goes through the `ela` ProxyJump (see ~/.ssh/config Host clariden).
set -euo pipefail

REMOTE=${REMOTE:-clariden}
SC=${SC:-/iopsstor/scratch/cscs/fffoivos}
REPO_ROOT=${REPO_ROOT:-$SC/repo/glossapi-tokenizer-extension}
LOCAL_DIR=$(cd "$(dirname "$0")/.." && pwd)            # .../15_clean_academic
REMOTE_DIR=$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic
# home→CSCS multiplexing breaks the ela ProxyJump; force a fresh channel each time.
SSHX="ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=30"

echo "=== rsync $LOCAL_DIR → $REMOTE:$REMOTE_DIR ==="
rsync -az --delete -e "$SSHX" \
  --exclude 'reference_detector/target/' --exclude 'out/' \
  --exclude '**/__pycache__/' --exclude 'investigation/_raw_synthesis.json' \
  "$LOCAL_DIR"/ "$REMOTE:$REMOTE_DIR"/

echo "=== build aarch64 binary on the LOGIN node (needs crates.io) ==="
# Feed the remote build script over stdin (heredoc) to a login shell; avoids the broken
# nested-quote form `bash -lc "'...'"` which the remote shell parsed as a single quoted token
# (started with a literal ' -> 'command not found', exit 127). $REMOTE_DIR expands locally;
# \$HOME stays literal for remote expansion.
$SSHX "$REMOTE" bash -ls <<EOF
  set -e
  command -v cargo >/dev/null || { echo bootstrapping rustup; curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; source \$HOME/.cargo/env; }
  source \$HOME/.cargo/env 2>/dev/null || true
  cd $REMOTE_DIR/reference_detector
  cargo build --release
  cargo test --release 2>&1 | grep "test result" | head -2
  ls -la target/release/reference_detect
EOF
echo "=== submit with: ssh $REMOTE 'cd $REMOTE_DIR/clariden && sbatch run_academic_refs.sbatch' ==="
