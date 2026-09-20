# test_overpass_endpoint_config.py is a standalone script (see its own
# docstring: "Run with: python test_overpass_endpoint_config.py"). It calls
# sys.exit() at module scope, which crashes pytest's collection if pytest
# tries to import it just because it matches the test_*.py glob.
collect_ignore = ["test_overpass_endpoint_config.py"]
