FROM nvcr.io/nvidia/pytorch:20.12-py3

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt