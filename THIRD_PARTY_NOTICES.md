# Third-Party Notices

This project includes and/or bundles third-party components. The notices below cover only artifacts
that are vendored into the repository or bundled into the standalone distribution output.

For transitive Python dependencies installed into `site-packages/` in a standalone build, consult
each package's metadata (e.g. `*.dist-info/`) for license details.

## Vendored JavaScript

### htmx
- Project: https://htmx.org/ (repo: https://github.com/bigskysoftware/htmx)
- Version: `1.9.12`
- License: MIT
- Source (downloaded): https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js
- Location in repo: `ragprep/web/static/htmx.min.js`

## Bundled runtime

### python-build-standalone (CPython runtime)
- Project: https://github.com/indygreg/python-build-standalone
- Used to download and bundle a standalone CPython runtime for the distribution.
- The exact tag/asset used is recorded in `BUILD_INFO.txt` inside the standalone output.
- Location in standalone output: `python/`

## Not bundled
- LightOnOCR model weights are downloaded at runtime (Hugging Face) and are not shipped inside the
  standalone artifact by default.

