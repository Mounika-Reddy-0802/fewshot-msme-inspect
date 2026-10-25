"""The project pages of the app render without errors from the committed records.

They need no datasets and no model, so they must work on any machine a faculty
member opens the app on. The live-demo pages load CLIP and are exercised by the
scripted rehearsal in docs/cross_verification.md instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["views/overview.py", "views/how_it_works.py", "views/results.py", "views/about.py"]


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("FEWSHOT_DATA_ROOT", str(tmp_path / "no-data"))
    monkeypatch.setenv("FEWSHOT_RESULTS_ROOT", str(tmp_path / "results"))
    return AppTest.from_file("app/demo.py", default_timeout=120).run()


@pytest.mark.parametrize("page", PAGES)
def test_project_page_renders(app, page):
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.title


@pytest.mark.parametrize("dataset", ["MVTec AD", "VisA", "NEU-DET"])
def test_results_page_renders_every_dataset(app, dataset):
    app.switch_page("views/results.py").run()
    [r for r in app.radio if r.label == "Dataset"][0].set_value(dataset).run()
    assert not app.exception, [e.value for e in app.exception]


def test_overview_reads_headline_numbers(app):
    values = [m.value for m in app.metric]
    assert values and "-" not in values[:4]
