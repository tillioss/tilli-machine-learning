FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update

RUN pip3 install joblib

RUN pip3 install numpy

RUN pip3 install flask

RUN pip3 install sklearn

RUN pip3 install -U scikit-learn scipy matplotlib

COPY . .

EXPOSE  5000

CMD [ "python3", "-u" , "server/server.py"]
