FROM tensorflow/tensorflow:2.4.2-gpu

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD ["main.py"]