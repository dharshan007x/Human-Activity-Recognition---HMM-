"""Automated Streamlit AppTest tests verifying error-free page loads."""

from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_without_exception() -> None:
    """Verify that app.py compiles, initializes navigation, and loads without unhandled exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App loaded with exception: {at.exception}"
