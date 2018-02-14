.PHONY: test
test:
	python3 -m unittest discover --start-directory smart_importer --pattern "*_test.py"

.PHONY: install
install:
	pip3 install --editable .

