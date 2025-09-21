FROM python:latest

WORKDIR /carreviewapp

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY car_review_LLM_model.trained.py .

CMD ["python", "car_review_LLM_model_trained.py"]