#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 2 ]] || { echo "usage: $0 LEAF_SWITCH MINIMUM_NODES" >&2; exit 2; }
leaf_switch=$1
minimum_nodes=$2
[[ "$leaf_switch" =~ ^group[0-9]+$ ]] || { echo "invalid leaf switch: $leaf_switch" >&2; exit 2; }
[[ "$minimum_nodes" =~ ^[1-9][0-9]*$ ]] || { echo "invalid minimum node count: $minimum_nodes" >&2; exit 2; }

topology=$(scontrol show topology)
leaf_expression=$(awk -v target="SwitchName=$leaf_switch" '
  $1 == target && $2 == "Level=0" {
    nodes=$4; sub(/^Nodes=/, "", nodes); print nodes
  }
' <<<"$topology")
global_expression=$(awk '
  $1 == "SwitchName=global" {
    nodes=$4; sub(/^Nodes=/, "", nodes); print nodes
  }
' <<<"$topology")
[[ -n "$leaf_expression" && -n "$global_expression" ]] || {
  echo "leaf switch is absent from Clariden topology: $leaf_switch" >&2
  exit 2
}

mapfile -t allowed_nodes < <(scontrol show hostnames "$leaf_expression")
(( ${#allowed_nodes[@]} >= minimum_nodes )) || {
  echo "$leaf_switch has ${#allowed_nodes[@]} nodes, fewer than required $minimum_nodes" >&2
  exit 2
}
declare -A allowed_set=()
for node in "${allowed_nodes[@]}"; do allowed_set["$node"]=1; done
excluded_nodes=()
while IFS= read -r node; do
  [[ -n "${allowed_set[$node]:-}" ]] || excluded_nodes+=("$node")
done < <(scontrol show hostnames "$global_expression")
(( ${#excluded_nodes[@]} > 0 )) || { echo "empty exclusion set" >&2; exit 2; }
excluded_csv=$(IFS=,; printf '%s' "${excluded_nodes[*]}")
scontrol show hostlist "$excluded_csv"
