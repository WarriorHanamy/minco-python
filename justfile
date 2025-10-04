set shell := ["bash", "-lc"]

# Incrementally rebuild the minco extension.
rebuild:
	uv run tools/rebuild_extension

# Regenerate CasADi flatness sources and headers.
build-casadi-flatness:
	uv run tools/build_casadi_flatness
