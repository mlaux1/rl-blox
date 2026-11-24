"""Run the doctests of the imported rl_blox package.

The doctests run on the imported modules instead of the source files in
the repository, so they work whether rl_blox is installed in editable mode
or as a regular package.
"""

import doctest
import importlib
import pkgutil

import pytest

import rl_blox

FLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS


def _modules_with_doctests():
    finder = doctest.DocTestFinder()
    names = []
    for info in pkgutil.walk_packages(rl_blox.__path__, "rl_blox."):
        module = importlib.import_module(info.name)
        if any(test.examples for test in finder.find(module)):
            names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("name", _modules_with_doctests())
def test_doctests(name):
    result = doctest.testmod(importlib.import_module(name), optionflags=FLAGS)
    assert result.failed == 0
