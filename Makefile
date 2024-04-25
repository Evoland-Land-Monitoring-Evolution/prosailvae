CONDA_ENV = prosailvae
CONDA_SRC = :
MODULE_PURGE = :
CONDA_ACT = :
ifneq ($(wildcard /work/CESBIO/*),)
	MODULE_PURGE:=module purge
	CONDA_SRC:=module load conda/22.11.1
	CONDA_ACT:=conda activate /work/scratch/env/$(USER)/virtualenv/$(CONDA_ENV)
endif
ifneq ($(wildcard /gpfsdswork/*),)
	MODULE_PURGE:=module purge
	CONDA_SRC:=module load anaconda-py3/2023.09
	CONDA_ACT:=conda activate $(CONDA_ENV)
endif
CONDA = $(MODULE_PURGE) && $(CONDA_SRC) && $(CONDA_ACT)
PYPATH = PYTHONPATH=./src/:${PYTHONPATH}

all: check test

#####################
# Environment setup #
#####################
.PHONY:
build_conda:
	bash create-conda-env.sh $(CONDA_ENV)

#######################
# Testing and linting #
#######################
.PHONY: check
check: tests pylint mypy

.PHONY: test
TESTARGS?=tests/  # default argument for the make test target
test:
	# you can use
	# make test TESTARGS="-k mytest"
	# make test TESTARGS="-m \"not marker\""
	$(CONDA) && pytest -vv $(TESTARGS)

test_no_slow:
	$(CONDA) && pytest -vv -m "not slow" tests/

test_no_hpc:
	$(CONDA) && pytest -vv -m "not hpc" tests/

test_no_hpc_one_fail:
	$(CONDA) && pytest -vv -x -m "not hpc" tests/

test_one_fail:
	$(CONDA) && pytest -vv -x tests/

test_one_fail_no_slow:
	$(CONDA) && pytest -vv -m "not slow" -x tests/



PYLINT_IGNORED = ""
#.PHONY:
pylint:
	-@$(CONDA) && echo "\npylint:\n======="  && pylint --ignore=$(PYLINT_IGNORED) src/ tests

#.PHONY:
perflint:
	-@$(CONDA) && echo "\nperflint:\n======="  && pylint --ignore=$(PYLINT_IGNORED) src/ tests  --load-plugins=perflint

#.PHONY:
ruff:
	-@$(CONDA) && echo "\nruff:\n=====" && ruff check . --output-format pylint

#.PHONY:
rufffix:
	-@$(CONDA) && echo "\nruff:\n=====" && ruff check . --fix --output-format pylint

#.PHONY:
mypy:
	-@$(CONDA) && echo "\nmypy:\n=====" && mypy src/ tests/

#.PHONY:
pyupgrade:
	-@$(CONDA) && echo "\npyupgrade:\n========="
	-@$(CONDA) && find ./src/ -type f -name "*.py" -print |xargs pyupgrade --py310-plus
	-@$(CONDA) && find ./tests/ -type f -name "*.py" -print |xargs pyupgrade --py310-plus

#.PHONY:
autowalrus:
	-@$(CONDA) && echo "\nautowalrus:\n==========="
	-@$(CONDA) && find ./src/ -type f -name "*.py" -print |xargs auto-walrus
	-@$(CONDA) && find ./tests/ -type f -name "*.py" -print |xargs auto-walrus

#.PHONY:
refurb:
	-@$(CONDA) && echo "\nrefurb:\n======="
	-@$(CONDA) && refurb tests/
	-@$(CONDA) && cd src/ && refurb prosailvae/

#.PHONY:
lint: pylint ruff mypy refurb

#.PHONY:
fix: rufffix pyupgrade autowalrus

#.PHONY:
#doc:
#	$(#CONDA) && sphinx-build docs docs/_build

#tox:
#	$(#CONDA) && tox
