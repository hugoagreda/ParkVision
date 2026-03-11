.PHONY: install api dashboard

install:
	pip install -r requirements.txt

api:
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py
