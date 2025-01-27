import os
from pathlib import Path

from invoke import task

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))


@task
def format(ctx):
    """Run black and isort"""
    for cmd in (
        "black . --line-length 120",
        "isort .",
    ):
        ctx.run(cmd, echo=True)


@task
def coverage(ctx):
    """Generate coverage reports"""
    for cmd in ("pytest -v --cov --cov-report term-missing",):
        ctx.run(cmd, echo=True)


@task
def test(ctx):
    """Run tests"""
    print("Testing...")
    for cmd in ("pytest -v --junitxml=build/pytest_tests.xml",):
        ctx.run(cmd, echo=True)
