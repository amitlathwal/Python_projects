FROM python:3
ENV PYTHONUNBUFFERED 1

WORKDIR /contain
ADD . /contain
COPY ./requirements.txt /contain/requirements.txt

RUN pip install -r requirements.txt
COPY . /contain

CMD ["python","weather/manage.py","runserver"]
