#!/bin/bash

# check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# kill process at specified port
kill_port() {
    local PORT=$1

    if [ -z "$PORT" ]; then
        echo "Usage: kill_port <port_number>"
        return 1
    fi

    local PID
    PID=$(lsof -ti tcp:$PORT)

    if [ -n "$PID" ]; then
        echo "Port $PORT is in use by process ID $PID. Killing it..."
        kill -9 $PID
        echo "Process $PID has been killed."
    else
        echo "Port $PORT is not in use."
    fi
}

# calculate unique(-ish) port number
string_to_port() {
  local input="$1"
  local hash
  local hash_prefix
  local hash_int
  local port_range=$((32767 - 1024 + 1))

  hash=$(echo -n "$input" | sha256sum | awk '{print $1}')
  hash_prefix=${hash:0:8}  # First 4 bytes (8 hex chars = 32 bits)
  hash_int=$((16#$hash_prefix))

  echo $((1024 + (hash_int % port_range)))
}


export container=deepracer
export image=deepracer


# check for Apptainer
if command_exists apptainer; then

    apptainer instance stop "$container" || echo "No ${container} instance running."

    overlay=/tmp/"$container"_overlay
    rm -rf "$overlay"

    my_port=$(string_to_port "$USER")

    echo "Stopped deepracer Apptainer container at port ${my_port}."

# check for Docker
elif command_exists docker; then
    
    docker stop "$container"

    my_port=8888

    echo "Stopped deepracer Docker container at port ${my_port}."

else

    # if neither Docker nor Apptainer is found
    echo "Neither Docker nor Apptainer is installed"

fi

sleep 2

# # just make sure nothing is running
# kill_port "$my_port"
echo "Killed process at port ${my_port}."