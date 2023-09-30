FROM osrf/ros:melodic-desktop-full

# Install GTSAM
RUN apt-get update \
    && apt install -y software-properties-common \
    && add-apt-repository -y ppa:borglab/gtsam-release-4.0 \
    && apt-get update \
    && apt install -y libgtsam-dev libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SLAM support library
RUN apt-get update \
    && apt-get install -y curl \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get install -y \
    ros-melodic-navigation \
    ros-melodic-robot-localization \
    ros-melodic-robot-state-publisher \
    ros-melodic-pcl-ros \
    libopencv-dev \
    libopencv-contrib-dev \
    libpcap-dev \
    python3-pip \
    git \
    unzip \
    ros-melodic-roslint \
    && rm -rf /var/lib/apt/lists/*

# Install ceres 
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev 
RUN git clone -b 2.1.0 https://ceres-solver.googlesource.com/ceres-solver /ceres-solver
WORKDIR /ceres-solver/build
RUN cmake .. && make -j$(nproc) && make install
RUN apt install vim -y

# Add packages
WORKDIR /catkin_ws 
ADD ./livox_ros_driver /catkin_ws/src/livox_ros_driver
RUN bash -c 'source /opt/ros/melodic/setup.bash && \
    catkin_make -DCATKIN_WHITELIST_PACKAGES="livox_ros_driver"'
ADD ./union-cloud  /catkin_ws/src/union-cloud
RUN bash -c 'source /opt/ros/melodic/setup.bash && \
    catkin_make -DCATKIN_WHITELIST_PACKAGES="union_cloud"'
ADD ./mm-loam  /catkin_ws/src/mm-loam
RUN bash -c 'source /opt/ros/melodic/setup.bash && \
    catkin_make -DCATKIN_WHITELIST_PACKAGES="mm_loam"'
COPY "entrypoint.sh" "/entrypoint.sh"
ENTRYPOINT [ "/entrypoint.sh" ]