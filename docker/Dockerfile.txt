FROM ufoym/deepo:pytorch
LABEL maintainer="gongzhiqiang@alumni.sjtu.edu.cn"

WORKDIR /tmp
COPY requirements.txt ./
RUN pip install -r requirements.txt