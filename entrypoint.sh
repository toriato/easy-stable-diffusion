#!/bin/bash
jupyter notebook \
  --allow-root \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --port_retries=0 \
  --token='' \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.disable_check_xsrf=True
