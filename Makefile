.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: test
test:
	python -m unittest discover -p "*_test.py"
