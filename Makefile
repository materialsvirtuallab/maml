## Helper for testing

PROJECT_NAME := maml

all: pycodestyle flake8 mypy pydocstyle

pycodestyle:
	echo "pycodestyle checks..."
	pycodestyle $(PROJECT_NAME)
	echo "--- Done ---"

flake8:
	echo "flake8 checks..."
	flake8 --count --show-source --statistics $(PROJECT_NAME)
    # exit-zero treats all errors as warnings.
	flake8 --count --exit-zero --max-complexity=20 --statistics $(PROJECT_NAME)
	echo "--- Done ---"

mypy:
	echo "mypy checks..."
	mypy $(PROJECT_NAME)
	echo "--- Done ---"

pydocstyle:
	echo "pydocstyle checks..."
	pydocstyle --count $(PROJECT_NAME)
	echo "--- Done ---"

pylint:
	echo "pylint checks..."
	pylint --exit-zero $(PROJECT_NAME)
	echo "--- Done ---"

test:
	echo "pytest checks..."
	pytest $(PROJECT_NAME)
	echo "--- Done ---"
