FROM osgeo/gdal:ubuntu-small-3.6.1

RUN apt update -y && \
	apt upgrade -y

RUN apt install -y python3-pip

ADD requirements.txt /
RUN pip3 install -r /requirements.txt 

RUN mkdir -p /inputdir && \
	mkdir -p /outputdir

ADD *.py /

#RUN apt autoremove --purge -y python3-pip
#RUN	rm -rf /var/lib/apt/lists/*