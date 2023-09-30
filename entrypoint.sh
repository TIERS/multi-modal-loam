#!/bin/bash
set -e

# Setup ros environment
source "/opt/ros/melodic/setup.bash" --

# Setup package
source "/catkin_ws/devel/setup.bash" --

exec "$@"