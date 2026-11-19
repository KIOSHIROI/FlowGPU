# Public test suite

Run:

```sh
python -m pytest -q
```

The suite is CPU-first: emitted Triton source is executed through a CPU Triton
double. Tests that depend on private competition or evidence fixtures are not part
of this public snapshot.
