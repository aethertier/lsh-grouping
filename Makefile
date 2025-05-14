PACKAGE_NAME = "lsh-grouping"

.PHONY: install uninstall

install:
	pip install --upgrade pip setuptools
	pip install -e .

uninstall:
	pip uninstall -y $(PACKAGE_NAME)
