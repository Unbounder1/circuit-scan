FROM python:3.13-slim

WORKDIR /api-circuitscan

COPY lib/* /api-circuitscan/lib/
COPY requirements.txt /api-circuitscan/
COPY main.py /api-circuitscan/

COPY 
