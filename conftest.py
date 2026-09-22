# These are standalone scripts, not pytest modules (see each one's docstring:
# "Run with: python <name>.py"). They call sys.exit() at module scope, which
# does not merely fail collection — it aborts the whole pytest run with an
# INTERNALERROR, so every other test in the repo silently stops running too.
#
# test_auth_tokens.py was added alongside the Supabase token verification and
# only matched the test_*.py glob by convention; nothing had imported it under
# pytest until the auth rewrite made it fail, which is when the abort surfaced.
#
# test_overpass_compose_persistence.py is another. Note it aborts the run on
# SUCCESS as well: it ends in sys.exit(0), and a SystemExit raised while pytest
# is importing a module stops collection whatever its code. So "all its checks
# passed" and "no tests collected in the whole repo" are the same run.
collect_ignore = [
    "test_overpass_endpoint_config.py",
    "test_auth_tokens.py",
    "test_overpass_compose_persistence.py",
]
