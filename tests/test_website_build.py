"""The public site builds from the repository's documents, and every link inside it leads somewhere."""
import pytest

pytest.importorskip("markdown_it")

from website import build  # noqa: E402


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build.build(tmp_path_factory.mktemp("site"))


def test_no_link_inside_the_site_is_broken(site):
    assert build.broken(site) == []


def test_the_main_pages_and_the_replay_are_there(site):
    for page, _ in build.NAV:
        assert (site.out / page).is_file()
    assert (site.out / "demo" / "index.html").is_file()
    assert (site.out / "results" / "training-2026-11-15" / "README.html").is_file()


def test_every_page_carries_the_prototype_notice(site):
    for page in site.written:
        assert "not therapy and not a crisis service" in (site.out / page).read_text(encoding="utf-8")


def test_no_page_links_to_the_private_repository_or_to_a_missing_file(site):
    for page in site.written:
        text = (site.out / page).read_text(encoding="utf-8")
        assert build.PRIVATE not in text, page
        assert 'href="missing:' not in text, page


def test_wording_that_only_fits_github_is_not_on_the_site(site):
    for page in ("how-it-works.html", "results.html"):
        text = (site.out / page).read_text(encoding="utf-8")
        assert "download and open locally" not in text
        assert "this branch" not in text


def test_every_page_has_a_title_an_icon_and_a_link_preview(site):
    for page in site.written:
        text = (site.out / page).read_text(encoding="utf-8")
        assert ".md ·" not in text.split("</title>")[0], page  # a file path is not a title
        assert 'rel="icon"' in text and f'og:url" content="{build.PUBLIC}/{page}"' in text, page
    assert (site.out / "static" / "favicon.svg").read_text(encoding="utf-8").startswith("<svg xmlns=")


def test_the_home_page_draws_the_run_from_its_records(site):
    home = (site.out / "index.html").read_text(encoding="utf-8")
    assert home.count("<title>Round ") == 10  # the ten rounds that reached the gate
    assert home.count('r="6.5"') == 10 and "gpt-6-luna" in home and "Qwen3.5-9B" in home


def test_the_build_deletes_only_an_earlier_build(tmp_path):
    (tmp_path / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="not an earlier build"):
        build.build(tmp_path)
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "mine"
    (tmp_path / ".nojekyll").write_text("", encoding="utf-8")  # another Pages site is not an earlier build
    with pytest.raises(ValueError, match="not an earlier build"):
        build.build(tmp_path)
    (tmp_path / build.MARKER).write_text("", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="not an earlier build"):
        build.build(tmp_path)


def test_a_second_build_replaces_the_first(site):
    again = build.build(site.out)
    assert again.written == site.written and not (site.out / "notes.txt").exists()


def test_a_bare_url_is_a_link_and_a_file_name_is_not():
    body = build.render("See https://arxiv.org/abs/2304.03442 and memory.md and docs/agents.md.")[0]
    assert '<a href="https://arxiv.org/abs/2304.03442">' in body
    assert body.count("<a ") == 1


def test_a_build_that_failed_part_way_can_be_run_again(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("a content check failed")

    with monkeypatch.context() as patch:
        patch.setattr(build.pages, "written", fail)
        with pytest.raises(ValueError, match="a content check failed"):
            build.build(tmp_path)
    assert any(tmp_path.iterdir())
    assert "index.html" in build.build(tmp_path).written
