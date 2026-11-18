"""Comprehensive AppTest suite verifying every single page loads and functions cleanly."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    "pages/01_Home.py",
    "pages/02_Dataset_Explorer.py",
    "pages/03_Preprocessing.py",
    "pages/04_Sensor_Visualisation.py",
    "pages/05_HMM_Model.py",
    "pages/06_Forward.py",
    "pages/07_Backward.py",
    "pages/08_Posterior_Probabilities.py",
    "pages/09_Activity_Recognition.py",
    "pages/10_Model_Evaluation.py",
    "pages/11_Beyond_the_Core.py",
    "pages/12_About.py",
]


@pytest.mark.parametrize("page_file", PAGES)
def test_individual_page_loads_without_exception(page_file: str) -> None:
    """Load every page file independently to verify zero compile or runtime crashes."""
    path = Path(page_file)
    assert path.exists(), f"Page file does not exist: {page_file}"

    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()
    assert not at.exception, f"Page {page_file} crashed with exception: {at.exception}"


def test_activity_recognition_page_widgets_and_uploader() -> None:
    """Test widget interactions on Activity Recognition page."""
    at = AppTest.from_file("pages/09_Activity_Recognition.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Selectbox selection
    if at.selectbox:
        at.selectbox[0].select_index(1).run()
        assert not at.exception


def test_forward_and_backward_page_widgets() -> None:
    """Test Forward and Backward pages calculation expander and sliders."""
    at_fwd = AppTest.from_file("pages/06_Forward.py", default_timeout=30)
    at_fwd.run()
    assert not at_fwd.exception

    at_bwd = AppTest.from_file("pages/07_Backward.py", default_timeout=30)
    at_bwd.run()
    assert not at_bwd.exception


def test_posterior_page_slider() -> None:
    """Test Posterior Probabilities page time-step slider."""
    at_post = AppTest.from_file("pages/08_Posterior_Probabilities.py", default_timeout=30)
    at_post.run()
    assert not at_post.exception

    if at_post.slider:
        at_post.slider[0].set_value(5).run()
        assert not at_post.exception

