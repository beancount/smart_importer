all:

.PHONY: test
test:
	tox -e py

.PHONY: lint
lint:
	pre-commit run -a
	tox -e lint

.PHONY: install
install:
	pip3 install --editable .

dist: smart_importer setup.cfg setup.py
	rm -rf dist
	python setup.py sdist bdist_wheel

# Before making a release, CHANGES needs to be updated and
# a tag and GitHub release should be created too.
.PHONY: upload
upload: dist
	twine upload dist/*
