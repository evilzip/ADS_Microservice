FROM python:3.11
WORKDIR /usr/src/app
COPY ./src ./
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
CMD streamlit run ads_app.py --server.port 8501