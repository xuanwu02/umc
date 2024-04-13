#! /bin/sh

env | sed "s/$USER/USER/g"
# set -x
# lsb_release -a
uname -a
lscpu || cat /proc/cpuinfo
cat /proc/meminfo
# inxi -F -c0
lsblk -a
# lsscsi -s
module list
# nvidia-smi
# (lshw -short -quiet -sanitize || lspci) | cat