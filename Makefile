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
	pip install --editable .

dist: smart_importer pyproject.toml
	rm -rf dist
	python -m build

# Before making a release, CHANGES needs to be updated and
# a tag and GitHub release should be created too.
.PHONY: upload
upload: dist
	twine upload dist/*
