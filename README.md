# ML Model Decorators

This repository contains code that shows how to create decorators for machine learning model instances.

This code is used in this [blog post](https://medium.com/p/f0be67973cc3).

## Requirements

- Python 3

## Installation 

The Makefile included with this project contains targets that help to automate several tasks.

To download the source code execute this command:

```bash
git clone https://github.com/schmidtbri/ml-model-decorators
```

Then create a virtual environment and activate it:

```bash
# go into the project directory
cd ml-model-decorators

make venv

source venv/bin/activate
```

Install the dependencies:

```bash
make dependencies
```

## Running the Unit Tests

To run the unit test suite execute these commands:

```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```