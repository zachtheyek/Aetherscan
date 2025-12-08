#!/bin/bash

# Get machine name and IP
HOSTNAME=$(hostname)
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Get output directory from first argument, default to current directory
OUTPUT_DIR="${1:-.}"

# Remove trailing slash if present
OUTPUT_DIR="${OUTPUT_DIR%/}"

# Output file path
OUTPUT_FILE="${OUTPUT_DIR}/${HOSTNAME}_system_info.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create the log file with system information
{
    echo "# System Info - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# Machine: $HOSTNAME"
    echo "# IP Address: $IP_ADDRESS"
    echo ""
    echo "## GPU Information"
    echo ""
    echo "### nvidia-smi: GPU status, driver version, CUDA version, memory usage, temperature, power"
    nvidia-smi
    echo ""
    echo "### nvcc --version: CUDA Toolkit version"
    if command -v nvcc &>/dev/null; then
        nvcc --version
    else
        echo "nvcc not found in PATH"
    fi
    echo ""
    echo "### cuDNN version: Deep learning primitives library version"
    if [ -f /usr/local/cuda/include/cudnn_version.h ]; then
        cat /usr/local/cuda/include/cudnn_version.h | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL"
    elif [ -f /usr/include/cudnn_version.h ]; then
        cat /usr/include/cudnn_version.h | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL"
    else
        echo "cuDNN version file not found"
    fi
    echo ""
    echo "## Operating System & Kernel Information"
    echo ""
    echo "### uname -a: Kernel version, architecture, hostname"
    uname -a
    echo ""
    echo "### /etc/os-release: Distribution name, version, ID"
    cat /etc/os-release
    echo ""
    echo "## CPU Information"
    echo ""
    echo "### lscpu: CPU architecture, cores, threads, cache, NUMA, flags"
    lscpu
    echo ""
    echo "## Memory Information"
    echo ""
    echo "### free -h: Total, used, free, and available RAM and swap"
    free -h
    echo ""
    echo "## Network Interfaces"
    echo ""
    echo "### ip addr: Network interfaces, IP addresses, MAC addresses"
    ip addr
    echo ""
    echo "### ip link: Network interface status and properties"
    ip link
    echo ""
    echo "## PCIe Topology"
    echo ""
    echo "### lspci: All PCI devices including GPUs, network cards, storage controllers"
    lspci
    echo ""
    echo "### lspci -tv: PCI device tree showing bus hierarchy"
    lspci -tv
    echo ""
    echo "## Storage Hardware"
    echo ""
    echo "### lsblk: Block devices, sizes, mount points, partition layout"
    lsblk
    echo ""
    echo "### lsblk -d -o name,size,model,serial: Detailed disk information"
    lsblk -d -o name,size,model,serial
    echo ""
    echo "## Temperature Sensors"
    echo ""
    echo "### sensors: CPU and system temperature readings"
    if command -v sensors &>/dev/null; then
        sensors
    else
        echo "lm-sensors not installed (install with: apt install lm-sensors)"
    fi
    echo ""
    echo "## Disk Usage"
    echo ""
    echo "### df -h: Filesystem disk space usage"
    df -h
} >"$OUTPUT_FILE"

echo "System information saved to $OUTPUT_FILE"
