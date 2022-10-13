FROM aeon/pytorch
ENV PUID=1001
ENV PGID=1001
ENV USERNAME=jupyter

# keep fresh :)
RUN apt update && apt upgrade -y 

# # update and add custom repository for latest python, old ubuntu suck :(
# RUN apt install -y software-properties-common &&\
#   add-apt-repository -y ppa:deadsnakes/ppa

# install dependencies for compile
RUN apt install -y \
  git curl \
  build-essential zlib1g-dev libgtk2.0-dev

# # install python 3.7.x (same as colab)
# RUN apt autoremove -y python2.7 python3 &&\
#   apt install -y python3.7 python3.7-distutils python3.7-dev &&\
#   ln -s $(which python3.7) /usr/local/bin/python

# # install pip
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# # install latest pytorch
# RUN pip install --extra-index-url https://download.pytorch.org/whl/cu116 torch torchvision

# install jupyterlab
RUN pip install jupyterlab

# create local user (root evil!)
RUN apt install sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN addgroup ${USERNAME} --gid ${PGID} &&\
  adduser ${USERNAME} --uid ${PUID} --gid ${PGID} &&\
  usermod -aG sudo ${USERNAME}

# copy entrypoint script
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

#----------------------------
# running from non-root user
#----------------------------
USER jupyter

# for Google Colab remote execution
RUN pip3 install --user ipywidgets jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 8888/tcp