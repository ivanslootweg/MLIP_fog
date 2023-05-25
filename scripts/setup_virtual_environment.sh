#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set variable to path of root of this project
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# We want to create the virtual environment in the scratch directory as /scratch/
# is a local disk (unique for each node) and therefore more performant.
#
# We make sure a valid directory to store virtual environments exists
# under the path /scratch/YOUR_USERNAME/virtual_environments
#
# If you call this script on your local computer (e.g, hostname != cn84, cn47 or cn48)
# the virtual environment will just be created in the root directory of this project.
if [[  "$HOSTNAME" == "slurm"* ]]; then
  echo "don't run this script on slurm22"
  return 1 2> /dev/null || exit 1
elif [[ "$HOSTNAME" != "cn"* ]]; then
  VENV_DIR=$PROJECT_DIR/venv
else
  VENV_DIR=/scratch/$USER/virtual_environments/tiny-voxceleb-venv
fi

echo "making the virtual environment at $VENV_DIR"

mkdir -p "$VENV_DIR"

# create the virtual environment
python3 -m venv "$VENV_DIR"

# create a symlink to the 'venv' folder if we're on the cluster
if [ ! -f "$PROJECT_DIR"/venv ]; then
  ln -sfn "$VENV_DIR" "$PROJECT_DIR"/venv
fi

# install the dependencies
source "$VENV_DIR"/bin/activate
PIP_NO_CACHE_DIR=off python3 -m pip install --upgrade pip
PIP_NO_CACHE_DIR=off python3 -m pip install wheel
PIP_NO_CACHE_DIR=off python3 -m pip install -r "$PROJECT_DIR"/requirements.txt
