#!/bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -C CPUs -M memory"
    echo -e "\t-C Maximum CPUs to allocate to the container, e.g. \"3\"."
    echo -e "\t-M Maximum memory to allocate to the container, e.g. \"6g\"."
    exit 1 # Exit script after printing help
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
done;

# assign default if empty
if [ -z "$cpus" ] || [ -z "$memory" ]
then
    cpus="${cpus:=3}"
    memory="${memory:=6g}"
    echo "Capping deepracer at ${cpus} CPUs and ${memory} memory.";
fi

source scripts/stop_deepracer.sh

source scripts/start_deepracer.sh \
    -C "$cpus" \
    -M "$memory" \
    -E "$evaluation" \
    -W "$world_name"
