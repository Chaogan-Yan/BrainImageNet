FROM tensorflow/tensorflow:1.13.1-py3-jupyter

MAINTAINER Chao-Gan Yan <ycg.yan@gmail.com>

RUN pip3 install -Iv keras==2.2.4
RUN pip3 install nibabel
RUN pip3 install matplotlib

COPY . .

ENTRYPOINT python3 y_Predict_Sex.py -i /in -o /out && python3 y_Predict_AD.py -i /in -o /out

