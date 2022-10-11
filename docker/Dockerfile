FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04

RUN apt update && apt upgrade -y
RUN apt install -y build-essential python3-dev python3-pip git
RUN pip3 install --upgrade pip

RUN apt install -y zlib1g-dev
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install numpy jupyterlab

# for Google Colab remote execution
RUN pip3 install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.allow_origin='https://colab.research.google.com'"]
EXPOSE 8080
