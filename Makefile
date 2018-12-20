all:

.PHONY: test
test:
	tox -e py

.PHONY: lint
lint:
	tox -e lint

.PHONY: install
install:
	pip3 install --editable .
