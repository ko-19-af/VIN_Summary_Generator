FROM python:latest

WORKDIR /carreviewapp

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY data_processing.py .

COPY car_review_LLM_Model.py .

COPY car_review_LLM_model.trained.py .

CMD ["python", "data_processing.py"]

CMD ["python", "car_review_LLM_Model.py"]

CMD ["python", "car_review_LLM_model_trained.py"]
