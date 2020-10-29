FROM python:3.7.2-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python ./imc_explicit.py