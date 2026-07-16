# Historical branching experiment

This directory preserves the former controller-based branching-size experiment
and its command-line wrapper.  Production branched Rao--Blackwell training does
not load these files: it uses the configured fixed beta table and ordinary
randomized rounding at every node, including the single distinguished root.

The Python files are outside the `pbg` package and pytest collection, so this
historical experiment is inert and not importable as an active implementation.
