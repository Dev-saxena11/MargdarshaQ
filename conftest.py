# These are standalone scripts, not pytest modules (see each one's docstring:
# "Run with: python <name>.py"). They call sys.exit() at module scope, which
# does not merely fail collection — it aborts the whole pytest run with an
# INTERNALERROR, so every other test in the repo silently stops running too.
#
# test_auth_tokens.py was added alongside the Supabase token verification and
# only matched the test_*.py glob by convention; nothing had imported it under
# pytest until the auth rewrite made it fail, which is when the abort surfaced.
#
# test_assistant_providers.py is the third of these. A standalone script is
# recognisable by its own docstring: "Run with: python <name>.py". If you add
# one, add it here too, or the whole suite stops running and the output says
# only "no tests ran" -- which names neither the file nor the reason.
collect_ignore = [
    "test_overpass_endpoint_config.py",
    "test_auth_tokens.py",
    "test_assistant_providers.py",
]
