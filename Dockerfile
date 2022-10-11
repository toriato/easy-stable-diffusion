FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04
ENV PUID=1001
ENV PGID=1001

RUN apt update && apt upgrade -y
RUN apt install -y build-essential python3-dev python3-pip git
RUN pip3 install --upgrade pip

RUN apt install -y zlib1g-dev
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install numpy jupyterlab

COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

# create local user (root evil!)
RUN addgroup jupyter --gid ${PGID}
RUN adduser jupyter --uid ${PUID} --gid ${PGID}
USER jupyter

# for Google Colab remote execution
RUN pip3 install --user jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 8888/tcp