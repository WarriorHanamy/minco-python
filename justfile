set shell := ["bash", "-lc"]

# Incrementally rebuild the minco extension.
rebuild:
	uv run tools/rebuild_extension
