FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04
ENV PUID=1001
ENV PGID=1001
ENV USERNAME=jupyter

RUN apt update && apt upgrade -y
RUN apt install -y build-essential python3.7 python3.7-dev python-pip git
RUN pip install --upgrade pip

RUN apt install -y zlib1g-dev
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install numpy jupyterlab

# create local user (root evil!)
RUN apt install sudo
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
RUN pip install --user jupyter_http_over_ws ipywidgets
RUN jupyter serverextension enable --py jupyter_http_over_ws

ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 8888/tcp