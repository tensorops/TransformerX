# Contributing to the TransformerX

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

Minor changes and improvements will be released on an ongoing basis. Larger
changes (e.g., changesets implementing a new paper) will be released on a
more periodic basis.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints (we recomend using Black plugin on Pycharm).

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Environment setup

We recommend using Pycharm which handles all these stuff for you with minimal interaction. Also, there are other good IDEs you can use.

otherwise:

```bash
name@user:~$ cd path/to/transformerX
name@user:path/to/transformerX~$ python3 -m venv myenv
name@user:path/to/transformerX~$ source myenv/bin/activate
(myenv) name@user:path/to/transformerX~$ pip3 install -r requirements-test.txt
```

## Coding Style

Two options to make sure that the code is formatted and linted properly:
* either you run black, mypy and isort before opening up your PR.

```bash
black .
isort . --profile black
flake8 --config .flake8
mypy --ignore-missing-imports --scripts-are-modules --pretty --exclude build/ --exclude stubs/ .
```

* or you can just install black plugin on Pycharm which will make sure that all of the above is run automatically anytime you commit.

After these steps each of your commits will run the same linting and formatting routines as the TransformerX continuous integration, which greatly helps getting your PRs all green !

## Testing

### Static analysis

```bash
mypy --ignore-missing-imports --scripts-are-modules --pretty --exclude stubs/ .
```

### Unit tests

```bash
pytest
```

or

``` bash
python -m pytest
```

### Check test coverage

``` bash
python -m pytest --cov-report term --cov=template  tests
```

## Commit Guidelines

We follow the same guidelines as AngularJS. Each commit message consists of a **header**,
a **body** and a **footer**.  The header has a special format that includes a **type**,
and a **subject**:

```bash
[<type>] <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on github as well as in various git tools.

### Type

Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **cleanup**: Changes that do not affect the meaning of the code (white-space, formatting, missing
  semi-colons, dead code removal etc.)
* **refactor**: A code change that neither fixes a bug or adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing tests or fixing them
* **chore**: Changes to the build process or auxiliary tools and libraries such as documentation
generation
* **docs**: Documentation only changes

## License

By contributing to *TransformerX*, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
