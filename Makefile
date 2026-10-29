.PHONY: install test doctor bench train generate serve clean

install:
	pip install -e .

test:
	python -m unittest discover -s tests -t .

doctor:
	python -m cloverforge doctor

bench:
	python -m cloverforge bench

train:
	python -m cloverforge train --skill 0.5

generate:
	python -m cloverforge generate --prompt clover24

serve:
	python -m cloverforge serve --port 8788

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
