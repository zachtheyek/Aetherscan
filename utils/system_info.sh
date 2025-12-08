#!/bin/bash

# Get machine name and IP
HOSTNAME=$(hostname)

# Output file
OUTPUT_FILE="${HOSTNAME}_system_info.log"
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Create the log file with system information
{
    echo "# System Info - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# Machine: $HOSTNAME"
    echo "# IP Address: $IP_ADDRESS"
    echo ""
    echo "## GPU Information"
    echo ""
    nvidia-smi
    echo ""
    echo "## CPU Information"
    echo ""
    lscpu
    echo ""
    echo "## Memory Information"
    echo ""
    free -h
    echo ""
    echo "## Disk Information"
    echo ""
    df -h
} >"$OUTPUT_FILE"

echo "System information saved to $OUTPUT_FILE"
