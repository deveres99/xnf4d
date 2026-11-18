# xnf4d
A stand-alone package for 4D normal form computation on one-turn-maps that is also compatible with xsuite.

## Installation

From the root of the source tree:

```bash
pip install .            # the normal form computation only (numpy, scipy, contourpy)
pip install ".[xsuite]"  # also one-turn maps, orbits and distributions of xsuite lines
```

Use `pip install -e ".[xsuite]"` for an editable install, in which changes to
the source code take effect without reinstalling. The optional dependencies
`examples` (adds matplotlib, for the scripts in `examples/`) and `test` (adds
pytest) are installed the same way, e.g. `pip install -e ".[test]"`.

## Tests

```bash
pytest
```
