#!/usr/bin/env bash
# Archive-only Capstor inode recovery for dated bibliography bootstrap outputs.
#
# This experiment-side recovery helper does not delete any source. It creates
# one archive on IOPS, compares it against the source tree, and writes a small
# verification marker. Deletion is deliberately a separate, post-verification
# operation.
set -euo pipefail

: "${CAPSTOR_SOURCE_ROOT:?set Capstor source root}"
: "${INODE_RECOVERY_ROOT:?set IOPS archive root}"

archive="${INODE_RECOVERY_ROOT}/bib_cleaning_bootstraps_20260727_28.tar"
partial="${archive}.partial"
log="${INODE_RECOVERY_ROOT}/archive.log"
marker="${INODE_RECOVERY_ROOT}/archive.verified"

roots=(
  bib_cleaning_bootstrap_0c6117f9_1cdb89b
  bib_cleaning_bootstrap_0c6117f9_284c120
  bib_cleaning_bootstrap_0c6117f9_88f06f5
  bib_cleaning_bootstrap_0c6117f9_e7f2f90
  bib_cleaning_bootstrap_0c6117f9_ffa4d72
  bib_cleaning_bootstrap_4adfdfd2_c86b1cb
  bib_cleaning_bootstrap_4adfdfd2_e7f2f90
  bib_cleaning_bootstrap_65a49d00_284c120
  bib_cleaning_bootstrap_698ca31c_284c120
  bib_cleaning_bootstrap_6f12e9d5_284c120
  bib_cleaning_bootstrap_74bd44dd_74a959e
  bib_cleaning_bootstrap_b8cc3ea2_284c120
  bib_cleaning_bootstrap_d262c4fd_284c120
  bib_cleaning_bootstrap_e8fbec2c_284c120
)

mkdir -p "${INODE_RECOVERY_ROOT}"
test ! -e "${archive}"
test ! -e "${partial}"
test ! -e "${marker}"
for root in "${roots[@]}"; do
  test -d "${CAPSTOR_SOURCE_ROOT}/${root}"
done

source_paths=()
for root in "${roots[@]}"; do
  source_paths+=("${CAPSTOR_SOURCE_ROOT}/${root}")
done

{
  printf 'started %s\n' "$(date -u +%FT%TZ)"
  printf 'source_entries '
  find "${source_paths[@]}" -xdev -printf . | wc -c
  printf 'source_bytes '
  find "${source_paths[@]}" -xdev -type f -printf '%s\n' | awk '{total += $1} END {print total + 0}'
  tar -cf "${partial}" -C "${CAPSTOR_SOURCE_ROOT}" "${roots[@]}"
  tar -df "${partial}" -C "${CAPSTOR_SOURCE_ROOT}" "${roots[@]}"
  mv "${partial}" "${archive}"
  printf 'archive_entries '
  tar -tf "${archive}" | wc -l
  sha256sum "${archive}"
  printf 'verified %s\n' "$(date -u +%FT%TZ)"
} >"${log}" 2>&1

touch "${marker}"
