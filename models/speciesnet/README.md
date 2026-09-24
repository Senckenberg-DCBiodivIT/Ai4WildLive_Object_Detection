# SpeciesNet (mirrored)

Not a Senckenberg model. This is a thin WildLive adapter around **SpeciesNet**, a
wildlife classifier built by Google.

- Upstream: [google/cameratrapai](https://github.com/google/cameratrapai)
- Author: Google Camera Traps AI team
- Licence: Apache-2.0
- Pinned here: `speciesnet==5.0.3`

Only the files in this folder are WildLive code. The model, its weights and its taxonomy
are Google's, installed from PyPI at build time — nothing upstream is copied into this
repository.

If SpeciesNet misidentifies an animal, report it
[upstream](https://github.com/google/cameratrapai). Open an issue here only for adapter
problems: box scaling, the score threshold, or name parsing.

## Why it differs from the template

Read [`template/`](../../template/) for the pattern to follow. This folder is the
exception:

- **No `classes.json`.** SpeciesNet's labels already carry the scientific name, so
  `map_result()` returns `scientificName` directly. It parses the genus and species out
  of SpeciesNet's `uuid;class;order;family;genus;species;common_name` string, and skips
  predictions with no species or no box.
- **Converts boxes.** SpeciesNet returns normalised `[x, y, width, height]`; WildLive
  wants pixel `[x1, y1, x2, y2]`.
- **Has a `preload.py`.** Weights download when the environment is built, not on the
  first request. Any model can use this hook.
- **Pins two transitive dependencies** for reasons unrelated to the model: `idna>=3.15`
  lifts a vulnerable version, and `setuptools<81` keeps a dependency working that still
  imports `pkg_resources`.
