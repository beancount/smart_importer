.PHONY: test
test:
	tox -e py3

.PHONY: lint
lint:
	tox -e lint

.PHONY: install
install:
	pip3 install --editable .

