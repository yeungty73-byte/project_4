#!/bin/bash


helpFunction()
{
    echo ""
    echo "Usage: $0 -C CPUs -M memory"
    echo -e "\t-C Maximum CPUs to allocate to the container, e.g. \"3\"."
    echo -e "\t-M Maximum memory to allocate to the container, e.g. \"6g\"."
    exit 1 # Exit script after printing help
}


# check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}


docker_image_exists() {
    local image="$1"
    if docker image inspect "$image" > /dev/null 2>&1; then
        echo "Image '$image' exists."
        return 0
    else
        echo "Image '$image' does not exist."
        return 1
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


# check if on a PACE ICE machine
on_pace_ice() {
    local var="$1"
    if [[ "$var" == *pace.gatech.edu ]]; then
        return 0  # Success (matches)
    else
        return 1  # Failure (does not match)
    fi
}


while getopts "C:M:E:W:" opt
do
    case "$opt" in
        C ) cpus="$OPTARG" ;;
        M ) memory="$OPTARG" ;;
        E ) evaluation="$OPTARG" ;;
        W ) world_name="$OPTARG" ;;
        ? ) helpFunction ;; # print helpFunction in case parameter is non-existent
    esac
done


# assign default if empty
if [ -z "$cpus" ] || [ -z "$memory" ]
then
    cpus="${cpus:=3}"
    memory="${memory:=6g}"
    echo "Capping deepracer at ${cpus} CPUs and ${memory} memory.";
fi

patches=patches
configs=configs

mkdir -p "$configs"
mkdir -p "$patches"

export base=uzairakbar/deepracer:v0
export container=deepracer
export image=deepracer


SCRATCH_DIR=''
if on_pace_ice "$HOSTNAME"; then
    SCRATCH_DIR="$HOME"/scratch
    
    # if [ -L "$HOME"/.conda ]; then
    #     echo "Conda already in scratch directory."
    # else
    #     mv "$HOME"/.conda "$SCRATCH_DIR"/.conda
    #     ln "$HOME"/.conda "$SCRATCH_DIR"/.conda
    #     echo "Moved conda to scratch directory."
    # fi
    
else
    SCRATCH_DIR="$PWD"
fi


# check for Apptainer
if command_exists apptainer; then
    echo "Building deepracer Apptainer container."

    apptainer pull deepracer_base.sif docker://"$base"

    yes no | apptainer build --ignore-fakeroot-command "$SCRATCH_DIR"/"$image".sif deepracer.def

    GYM_PORT=$(string_to_port "$USER")
    echo "Using port $GYM_PORT for deepracer."
    
    GAZEBO_PORT=$(string_to_port "GAZEBO_$USER")        # default is 11345
    GAZEBO_MASTER_URI="http://localhost:$GAZEBO_PORT"
    echo "Using port $GAZEBO_MASTER_URI for Gazebo Master."

    ROS_PORT=$(string_to_port "ROS_$USER")              # defaults is 11311
    ROS_MASTER_URI="http://localhost:$ROS_PORT"
    echo "Using port $ROS_MASTER_URI for ROS Master."

    overlay=/tmp/"$container"_overlay
    rm -rf "$overlay" && mkdir "$overlay"
    apptainer instance run \
        --no-mount "$HOME",/tmp,/dev,/etc/hosts,/etc/localtime,/proc,/sys,/var/tmp \
        --bind configs:/configs \
        --overlay "$overlay"/:/. \
        --env EVALUATION="$evaluation",EVAL_WORLD_NAME="$world_name",GYM_PORT="$GYM_PORT",GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI",ROS_MASTER_URI="$ROS_MASTER_URI" \
        "$SCRATCH_DIR"/"$image".sif "$container" \
        --cpus="$cpus" --memory="$memory"

    echo "Started deepracer Apptainer container."
    
# check for Docker
elif command_exists docker; then
    echo "Building deepracer Docker container."
    
    # pull base image
    if docker_image_exists "$base"; then
        echo "Docker image '$base' already exists. Skipping pull."
    else
        echo "Docker image '$base' not found. Pulling now..."
        docker pull "$base"
    fi

    # build P4 deepracer image
    if docker_image_exists "${image}:latest"; then
        echo "Docker image '$image' already exists. Skipping build."
    else
        echo "Docker image '$image' not found. Building now..."
        docker build -t "$image" .
        
        # prune just in case of dangling images
        docker system prune --force
    fi

    echo "Using port 8888 for deepracer."

    docker run --rm --detach \
        --name="$container" \
        -v "$PWD"/"$configs":/"$configs":ro \
        -p 8888:8888 \
        -e EVALUATION="$evaluation" \
        -e EVAL_WORLD_NAME="$world_name" \
        --cpus="$cpus" --memory="$memory" \
        "$image"
    
    echo "Started deepracer Docker container."
else
    # if neither Docker nor Apptainer is found
    echo "Neither Docker nor Apptainer is installed"
fi

sleep 2