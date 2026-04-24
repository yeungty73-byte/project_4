#!/bin/bash

# check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
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

export conda_env=deepracer

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

    rm -f "$SCRATCH_DIR"/"$image".sif
    overlay=/tmp/"$container"_overlay
    rm -rf "$overlay"

    echo "Cleaned deepracer Apptainer environment."

# check for Docker
elif command_exists docker; then
    
    docker rm "$container"

    docker image rm -f "$image"

    docker image rm -f "$base"

    docker system prune --force

    echo "Cleaned deepracer Docker environment."
fi

# check for Conda
if command_exists conda; then

    conda activate base
    conda remove --name "$conda_env" --all --yes --force

    echo "Cleaned deepracer Conda environment."
fi

# no environment found to clean
echo "Nothing more to clean!"