"""Independent verification scripts for the analytics/ math reference.

Each script in this package re-derives a quantity from the math files using an
independent reference implementation (scipy where possible, hand-coded numpy
otherwise) and asserts agreement with the rl_signaling implementation.

Run with, e.g.:

    .venv/bin/python -m analytics.scripts.verify_information_theory
"""
