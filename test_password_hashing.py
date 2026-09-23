"""
test_password_hashing.py
-------------------------
Signup hashes a password before it writes a user, so if hashing raises, every
registration returns a bare 500 — and because an unhandled exception is turned
into a response outside Starlette's CORS middleware, that 500 carries no
Access-Control-Allow-Origin header. The browser then reports a CORS failure,
which sends you looking at CORS_ORIGINS instead of at the password hash.

That is exactly what a routine rebuild caused: requirements.txt asked for
passlib[bcrypt] with no upper bound, pip took bcrypt 5.0.0, and passlib 1.7.4
(the last release, from 2020) reads bcrypt.__about__.__version__, which bcrypt
removed in 4.1. Unable to read the version, passlib cannot run its
wraparound-bug probe, and the probe's failure surfaces on the first hash of any
password as "password cannot be longer than 72 bytes" — whatever its length.

These run offline and need no database.
"""

import pytest

from app.api.routes import get_password_hash, verify_password


def test_a_password_can_be_hashed():
    """The failure this file exists for: hashing raised for every password."""
    assert get_password_hash("hunter2").startswith("$2")


def test_a_correct_password_verifies():
    assert verify_password("hunter2", get_password_hash("hunter2"))


def test_a_wrong_password_does_not_verify():
    assert not verify_password("not-the-password", get_password_hash("hunter2"))


def test_the_same_password_hashes_differently_each_time():
    """A per-hash salt, so identical passwords do not share a hash."""
    assert get_password_hash("hunter2") != get_password_hash("hunter2")


@pytest.mark.parametrize("password", [
    "a",                     # short
    "x" * 71,                # just under bcrypt's 72-byte input limit
    "x" * 72,                # exactly at it
    "pässwörd-ünicode",      # multi-byte characters
    "has spaces and $ymb0ls!",
])
def test_passwords_of_various_shapes_round_trip(password):
    assert verify_password(password, get_password_hash(password))


def test_the_installed_bcrypt_is_one_passlib_can_read():
    """
    passlib traps the version read and carries on, so an incompatible pair is
    silent until the first hash. Assert the thing that actually has to hold.
    """
    import bcrypt
    assert hasattr(bcrypt, "__about__"), (
        f"bcrypt {bcrypt.__version__} has no __about__; passlib 1.7.4 needs it. "
        "Hold bcrypt below 4.1, or move off passlib."
    )
