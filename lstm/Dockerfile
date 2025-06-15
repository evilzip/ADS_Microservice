FROM nvidia/cuda:12.4.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y nvidia-cuda-toolkit
FROM python:3.9
WORKDIR /usr/src/app
COPY ./src ./
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ,"--reload"]