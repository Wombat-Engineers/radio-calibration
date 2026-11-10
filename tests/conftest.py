"""Shared pytest setup."""

import matplotlib

# Tests must not open a window.
matplotlib.use("Agg")
