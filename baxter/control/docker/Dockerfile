# ---------------------------------------------------------------------------
# ROS Indigo + Baxter SDK 1.2.0, the last combination Rethink shipped.
# Ubuntu 14.04 and the Indigo debs are both long EOL, so the apt sources here
# are frozen archives. Don't "modernise" them -- there is nothing newer.
# ---------------------------------------------------------------------------
FROM ros:indigo-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# The base image still points apt at archive.ubuntu.com (which does serve
# trusty) and ROS at snapshots.ros.org (the frozen Indigo debs -- the live
# packages.ros.org dropped Indigo at EOL). Two things need fixing: the
# ubuntu-advantage ESM lists need credentials we don't have, and the ROS
# snapshot's signing key has expired.
RUN rm -f /etc/apt/sources.list.d/ubuntu-esm-infra.list \
          /etc/apt/sources.list.d/ubuntu-advantage*.list \
 && sed -i '/esm\.ubuntu\.com/d' /etc/apt/sources.list.d/*.list \
 && echo 'deb [trusted=yes] http://snapshots.ros.org/indigo/final/ubuntu trusty main' \
      > /etc/apt/sources.list.d/ros-latest.list \
 && echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid \
 && apt-get update

RUN apt-get install -y --force-yes --no-install-recommends \
      git python-pip python-dev build-essential \
      iputils-ping iproute2 net-tools curl vim less \
      python-wstool python-rosinstall python-catkin-tools \
      ros-indigo-control-msgs ros-indigo-trajectory-msgs \
      ros-indigo-cv-bridge ros-indigo-image-transport \
      ros-indigo-tf ros-indigo-tf2-ros ros-indigo-xacro \
      ros-indigo-joint-state-publisher ros-indigo-robot-state-publisher \
      ros-indigo-dynamic-reconfigure ros-indigo-joy \
 && rm -rf /var/lib/apt/lists/*

# Baxter SDK, pinned to the 1.2.0 release tags. Building it into the image
# keeps your own workspace clean -- it overlays this one.
ENV SDK_WS=/opt/baxter_ws
RUN mkdir -p $SDK_WS/src && cd $SDK_WS/src \
 && for r in baxter baxter_interface baxter_tools baxter_common baxter_examples; do \
      git clone -q --depth 1 -b v1.2.0 https://github.com/RethinkRobotics/$r.git; \
    done \
 && source /opt/ros/indigo/setup.bash \
 && cd $SDK_WS && catkin_make -j$(nproc) \
 && cd $SDK_WS && catkin_make install

# rosbridge: the JSON/websocket bridge our Python-3 brain (LLM planner,
# Dream-1 client, perception) uses to reach Baxter without importing the
# Python-2 SDK. Appended as its own layer so it does not invalidate the
# expensive SDK build cache above. Available from the frozen Indigo snapshot.
RUN apt-get update \
 && apt-get install -y --force-yes --no-install-recommends \
      ros-indigo-rosbridge-suite \
 && rm -rf /var/lib/apt/lists/*

# The upstream baxter.sh, kept pristine at /opt. The entrypoint stamps your
# hostname/IP into a copy in the workspace at container start.
RUN cp $SDK_WS/src/baxter/baxter.sh /opt/baxter.sh.template

COPY scripts/entrypoint.sh /opt/entrypoint.sh
COPY scripts/baxter-connect.sh /opt/baxter-connect.sh
RUN chmod +x /opt/entrypoint.sh /opt/baxter-connect.sh

RUN echo 'source /opt/ros/indigo/setup.bash' >> /root/.bashrc \
 && echo 'source /opt/baxter_ws/devel/setup.bash' >> /root/.bashrc \
 && echo 'cd /root/ros_ws' >> /root/.bashrc

WORKDIR /root/ros_ws
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["bash"]
