FROM python:3.10-slim-buster

ENV VIRTUAL_ENV=venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY . .
RUN pip install -r requirements.txt
CMD ["python","main.py"]