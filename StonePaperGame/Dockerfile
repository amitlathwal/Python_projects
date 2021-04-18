FROM python:3

COPY requirements.txt .
ADD list.csv .
ADD Rem_icon.ico .
ADD main.py .

RUN pip install -r requirements.txt
CMD ["python","main.py"]