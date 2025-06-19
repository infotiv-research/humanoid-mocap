
FROM althack/ros2:humble-cuda-gazebo-nvidia

ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib
ENV PATH=/opt/ros/humble/bin:$PATH
ENV PYTHONPATH=/opt/ros/humble/lib/python3.8/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2

ENV DEBIAN_FRONTEND=noninteractive

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QT_X11_NO_MITSHM=1

# To fix the expired key issue
RUN apt-get update && \
    apt-get install -y curl gnupg2 lsb-release && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key && \
    gpg --batch --yes --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg /tmp/ros.key && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list && \
    rm /tmp/ros.key

    
RUN apt-get update && apt-get install -y \
  # opengl stuff
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  x11-apps \
  mesa-utils \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6 \
  # other stuff
  bash-completion \
  build-essential \
  cmake \
  gdb \
  git \
  tig \
  # pylint3 \
  python3-argcomplete \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  vim \
  wget \
  gdbserver \
  htop \
  xterm \
  # Install ros distro testing packages
  ros-humble-ament-lint \
  ros-humble-launch-testing \
  ros-humble-launch-testing-ament-cmake \
  ros-humble-launch-testing-ros \
  ros-humble-gazebo-ros-pkgs \
  iputils-ping net-tools xvfb \
  ros-humble-usb-cam \
  python3-autopep8 \
  ros-humble-tf-transformations

RUN apt-get update && apt-get install -y \
    ros-humble-joint-state-publisher-gui ros-humble-moveit ros-humble-moveit-servo ros-humble-ign-ros2-control

# Install palanteer
RUN apt-get update && apt-get install -y \
    libdw-dev libunwind-dev     libgl1-mesa-glx libgl1-mesa-dri mesa-utils     && rm -rf /var/likb/apt/lists/*

RUN cd /opt && git clone https://github.com/dfeneyrou/palanteer && \
    cd palanteer && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# Install PIP requirements
COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt

# Install supl dependencies
RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends \
    gcc cmake libssl-dev make ninja-build ranger \
    && rm -rf /var/lib/apt/lists/*


RUN rm -rf /var/lib/apt/lists/*
RUN rosdep init || echo "rosdep already initialized"

RUN mkdir -p /opt/dependencies_ws/src
# Build ROS 2 source dependencies

# cd /opt/dependencies_ws
WORKDIR /opt/dependencies_ws
COPY ros_dependencies.repos ros_dependencies.repos
RUN vcs import src < ros_dependencies.repos
RUN apt-get update && \
    rosdep update && \
    rosdep install -q -y --from-paths src --ignore-src && \
    rm -rf /var/likb/apt/lists/*
#  building depoendencies
RUN . /opt/ros/humble/setup.sh && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release

WORKDIR /home/ros/src/


RUN echo "if [ -f /home/ros/install/setup.bash ]; then source /home/ros/install/setup.bash; fi" >> /home/ros/.bashrc
RUN echo "if [ -f /usr/share/gazebo/setup.sh ]; then . /usr/share/gazebo/setup.sh; fi" >> /home/ros/.bashrc
RUN echo "export RCUTILS_COLORIZED_OUTPUT=1" >> /home/ros/.bashrc
RUN echo "export GAZEBO_MASTER_URI=http://0.0.0.0:12345" >> /home/ros/.bashrc
RUN echo "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" >> /home/ros/.bashrc

COPY . /home/ros/src/
RUN chown -R ros:ros  /home/ros/src/
RUN usermod -aG video ros


# ONLY IF pymoveit2 pip installation doesn't work try these commands to install it manually

# git clone https://github.com/AndrejOrsula/pymoveit2.git
# rosdep install -y -r -i --rosdistro ${ROS_DISTRO} --from-paths .

RUN mkdir -p /usr/local/lib/python3.10/dist-packages/mediapipe/modules/pose_landmark
RUN chmod 777 /usr/local/lib/python3.10/dist-packages/mediapipe/modules/pose_landmark

USER ros


