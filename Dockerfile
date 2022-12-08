FROM python:3.9 

ADD clustering.py .

CMD [“python”, “./clustering.py”] 
