.PHONY: test
test:
	python -m unittest discover -p "*_test.py"

.PHONY: install
install:
	pip install -r requirements.txt

