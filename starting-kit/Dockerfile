FROM python:3.9.15

RUN mkdir -p /toy_dataset_flair-one && \
	mkdir -p /metadata


ADD py_module /py_module

ADD requirements.txt /
RUN pip install -r /requirements.txt


