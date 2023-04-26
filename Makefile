.PHONY: all mintscore abstractive_constraints test

all: mintscore abstractive_constraints
	@echo Done

mintscore:
	cd mintscore && pip install -e .

abstractive_constraints:
	cd abstractive_constraints && pip install -e .

test:
	pytest --doctest-modules

clean:
	rm -rf .pytest_cache
	rm -rf abstractive_constraints/.pytest_cache
	rm -rf abstractive_constraints/__pycache__
	rm -rf abstractive_constraints/abstractive_constraints.egg-info
	rm -rf abstractive_constraints/abstractive_constraints/__pycache__
	rm -rf abstractive_constraints/build
	rm -rf mintlcs/build
	rm -rf mintlcs/mintlcs.egg-info
	rm -rf mintlcs/var
	rm -rf mintscore/.pytest_cache
	rm -rf mintscore/mintscore.egg-info
	rm -rf mintscore/mintscore/__pycache__
	rm -rf misc/__pycache__/
