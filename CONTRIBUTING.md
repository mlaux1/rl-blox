# RL-BLOX Contribution Guide

Everyone is welcome to contribute.

There are several ways to contribute to rl-blox: you could

* send a bug report to the
  [bug tracker](http://github.com/mlaux1/rl-blox/issues)
* work on one of the reported issues
* write documentation
* add a new feature
* add tests
* add an example

## How to Contribute Code

This text is shamelessly copied from
[scikit-learn's](https://scikit-learn.org/stable/developers/contributing.html)
contribution guidelines.


The preferred way to contribute to rl-blox is to fork the main repository on GitHub, then submit a “pull request” (PR).

In the first few steps, we explain how to locally install rl-blox, and how to set up your git repository:

Create an account on GitHub if you do not already have one.

Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see this guide.

Clone your fork of the rl-blox repo from your GitHub account to your local disk:

```sh
git clone https://github.com/YourLogin/rl-blox.git  # add --depth 1 if your connection is slow
cd rl-blox
```

Build rl-blox from source as described in the README.md and return to this document.

Install the development dependencies:


```sh
pip install "rl-blox[dev]"
```

Add the upstream remote. This saves a reference to the main rl-blox repository, which you can use to keep your repository synchronized with the latest changes:

```sh
git remote add upstream https://github.com/mlaux1/rl-blox.git
```

Check that the upstream and origin remote aliases are configured correctly by running:

```sh
git remote -v
```

This should display:

```sh
origin    https://github.com/YourLogin/rl-blox.git (fetch)
origin    https://github.com/YourLogin/rl-blox.git (push)
upstream  https://github.com/mlaux1/rl-blox.git (fetch)
upstream  https://github.com/mlaux1/rl-blox.git (push)
```

You should now have a working installation of rl-blox, and your git repository properly configured. It could be useful to run some test to verify your installation. Please refer to useful pytest aliases and flags for examples.

The next steps now describe the process of modifying code and submitting a PR:

Synchronize your main branch with the upstream/main branch, more details on GitHub Docs:

```sh
git checkout main
git fetch upstream
git merge upstream/main
```

Create a feature branch to hold your development changes:

```sh
git checkout -b my_feature
```

and start making changes. Always use a feature branch. It’s good practice to never work on the main branch!

(Optional) Install pre-commit to run code style checks before each commit:

```sh
pip install pre-commit
pre-commit install
```

pre-commit checks can be disabled for a particular commit with git commit -n.

Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit:

```sh
git add modified_files
git commit
```

to record your changes in Git, then push the changes to your GitHub account with:

```sh
git push -u origin my_feature
```

Follow these instructions to create a pull request from your fork. This will send a notification to potential reviewers. You may want to consider sending a message to the discord in the development channel for more visibility if your pull request does not receive attention after a couple of days (instant replies are not guaranteed though).

It is often helpful to keep your local feature branch synchronized with the latest changes of the main rl-blox repository:

```sh
git fetch upstream
git merge upstream/main
```

Subsequently, you might need to resolve the conflicts. You can refer to the Git documentation related to resolving merge conflict using the command line.

### Learning Git

The Git documentation and http://try.github.io are excellent resources to get started with git, and understanding all of the commands shown here.

## Adding new algorithm or other features

Adding a new feature to rl-blox requires a few other changes:

* Please write [useful commit messages](https://cbea.ms/git-commit/)
* New classes or functions that are part of the public interface must be
  documented. We use [NumPy's conventions for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
* For new algorithms, a new example script should be added to the examples directory.
* Tests: Unit tests for new features are mandatory. They should cover all
  branches. Exceptions are plotting functions, debug outputs, etc. These
  are usually hard to test and are not a fundamental part of the library.

## Merge Policy

Usually, it is not possible to push directly to the develop or main branch for
anyone. Only tiny changes, urgent bugfixes, and maintenance commits can be
pushed directly to the develop branch by the maintainer without a review.
"Tiny" means backwards compatibility is mandatory and all tests must succeed.
No new feature must be added.

Developers have to submit pull requests. Those will be reviewed and merged by
a maintainer. New features must be documented and tested. Breaking changes must
be discussed and announced in advance with deprecation warnings.

## Versioning

Semantic versioning is used, that is, the major version number will be
incremented when the API changes in a backwards incompatible way, the
minor version will be incremented when new functionality is added in a
backwards compatible manner.
