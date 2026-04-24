#!/usr/bin/env bash
# v202b: Auto-discover the markov.rollout_worker's ZMQ REP port and export GYM_PORT.
# Source me AFTER start_deepracer.sh has come up.
#
# Strategy:
#   1) Find PIDs of `python .* markov.rollout_worker` owned by this user.
#   2) Ask `ss -ltnp` for TCP LISTEN ports bound to those PIDs.
#   3) If exactly one port is found, export GYM_PORT to it.
#   4) Fallback to the user-hashed port (matching start_deepracer.sh) if discovery fails.

string_to_port() {
  local hash hash_prefix hash_int port_range
  port_range=$((32767 - 1024 + 1))
  hash=$(echo -n "$1" | sha256sum | awk '{print $1}')
  hash_prefix=${hash:0:8}
  hash_int=$((16#${hash_prefix}))
  echo $((1024 + hash_int % port_range))
}

discover_rollout_port() {
  local pids port_list
  pids=$(pgrep -u "$USER" -f 'markov.rollout_worker' | tr '\n' '|' | sed 's/|$//')
  if [ -z "$pids" ]; then
    echo "" ; return 1
  fi
  # Extract :PORT from lines whose users: tuple contains one of our pids
  port_list=$(ss -ltnp 2>/dev/null \
    | awk -v pids="$pids" 'BEGIN{split(pids,P,"|")} {for(i in P) if(index($0,"pid="P[i])>0) print $4}' \
    | awk -F':' '{print $NF}' | sort -u)
  # Prefer the one that is NOT the ROS master or gazebo (those are large random); heuristic: smallest port
  echo "$port_list" | head -1
}

export GYM_HOST=127.0.0.1
export GAZEBO_PORT=$(string_to_port "GAZEBO_$USER")
export ROS_PORT=$(string_to_port "ROS_$USER")

_discovered=$(discover_rollout_port)
if [ -n "$_discovered" ]; then
  export GYM_PORT="$_discovered"
  echo "[env_for_client v202b] auto-discovered GYM_PORT=$GYM_PORT (from running rollout_worker)"
else
  export GYM_PORT=$(string_to_port "$USER")
  echo "[env_for_client v202b] no rollout_worker found; falling back to hashed GYM_PORT=$GYM_PORT"
fi
echo "[env_for_client v202b] GYM_HOST=$GYM_HOST GAZEBO_PORT=$GAZEBO_PORT ROS_PORT=$ROS_PORT"
