# Contributing

Thank you for your interest in scFoundry.

**Bug reports, fixes and documentation corrections** are welcome at any time as issues or
pull requests. Keep a pull request to one change, make sure the unit tests pass
(`python -m unittest tests/test_registry.py tests/test_launcher.py`), and for pipeline
changes include a demo run showing the effect.

**New methods** are accepted as pull requests once the accompanying paper is published and
the first public release is out. A method pull request must contain a `Dockerfile` under
`containers/<id>/`, the official checkpoint download, the module, the registry entry, a
filled-in method card and demo evidence. The full checklist, the module contracts and the
review criteria are in the documentation:

- Adding a method: https://svvord.github.io/scFM-eval-docs/extending/new-method.html
- Contributing a method: https://svvord.github.io/scFM-eval-docs/extending/contributing.html

Methods are reviewed against their authors' published recipe. A method that runs with its
authors' defaults, from their official weights, in an image built from a committed
`Dockerfile`, is what the benchmark promises — please help us keep that promise.
