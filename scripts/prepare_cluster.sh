#! /usr/bin/env bash
set -e

# only run this script on cn84
if [[ "$HOSTNAME" != "cn84" ]]; then
  echo "prepare_cluster.sh should only be run on cn84"
fi

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
CEPH_USER_DIR=/ceph/csedu-scratch/course/IMC030_MLIP/users/"$USER"
mkdir -p "$CEPH_USER_DIR"/slurm
chmod 700 "$CEPH_USER_DIR" # only you can access
ln -sfn "$CEPH_USER_DIR" "$SCRIPT_DIR"/../logs

# # make a symlink to the data in order to directly access it from the root of the project
# ln -sfn /ceph/csedu-scratch/course/IMC030_MLIP/data "$SCRIPT_DIR"/../data

# if .cache or .local is a symlink, remove it
# this is temporary for students of MLIP 2023
if [[ -L "$HOME/.cache" ]]; then
  rm "$HOME"/.cache
fi
if [[ -L "$HOME/.local" ]]; then
  rm "$HOME"/.local
fi

# make sure pip doesn't cache results
if ! grep -q "export PIP_NO_CACHE_DIR=" ~/.profile ; then
{
echo ""
echo "### disable pip caching downloads"
echo "export PIP_NO_CACHE_DIR=off"
} >> ~/.profile
fi

# set up a virtual environment located at
# /scratch/$USER/virtual_environments/tiny-voxceleb-venv
# and make a symlink to the virtual environment
# at the root directory of this project called "venv"
# uncomment this if you need a virtual environment on cn84
# ./setup_virtual_environment.sh

# make sure that there's also a virtual environment
# on the GPU nodes
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
srun -p csedu-prio -A cseduimc030 -q csedu-small -w cn47 ./setup_virtual_environment.sh

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
srun -p csedu-prio -A cseduimc030 -q csedu-small -w cn48 ./setup_virtual_environment.sh
