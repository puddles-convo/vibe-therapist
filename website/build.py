"""Build the public site into website/_site from the repository's own documents and records. No model, no network.

    python -m website.build              # write website/_site
    python -m website.build --out DIR    # write somewhere else

Needs markdown-it-py and linkify-it-py (website/requirements.txt). Text pages are the repository's Markdown files,
rendered; the hand-written pages are in website/pages.py, and every number they quote is checked against the file it
comes from.
"""
from __future__ import annotations

import argparse
import html
import os
import posixpath
import re
import shutil
from pathlib import Path

from markdown_it import MarkdownIt

from demo import build as demo
from website import pages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website" / "_site"
REPO = "https://github.com/puddles-convo/vibe-therapist"
PUBLIC = "https://puddles-convo.github.io/vibe-therapist"  # where GitHub Pages serves the site; link previews need it
PRIVATE = "https://github.com/puddles-convo/puddles-stack"  # a private repository: visitors get a 404, so no link
# Wording in README.md that is only right on GitHub, and what the site says in its place.
ON_SITE = ((" (download and open locally)", ""),
           ("this branch also implements", "the current code also implements"),
           ("which this branch now implements", "which the current code now implements"),
           ("is now implemented on this branch", "is now implemented"))
MARKDOWN = ("docs/**/*.md", "results/**/*.md", "figures/*.md", "policies/**/*.md")
COPIED = ("demo/index.html", "figures/fig1-player.html", "figures/fig2-player.html", "docs/media/puddles-robot.png")
ASSET = {".html", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
NAV = (("index.html", "Home"), ("vision.html", "Vision"), ("how-it-works.html", "How it works"),
       ("results.html", "Results"), ("runs.html", "Runs"), ("demo.html", "Demo"), ("docs.html", "Docs"),
       ("limits.html", "Limits"))
# README.md is split over these pages, by its level-2 headings.
README_PAGES = (("how-it-works.html", "How it works", ("How it learns", "The live conversation")),
                ("results.html", "Results", ("Results",)),
                ("design.html", "The design in depth", ("The design in depth",)),
                ("run-it.html", "Run it yourself", ("Quick start",)))

CURRENT = ' aria-current="page"'
MD = MarkdownIt("commonmark", {"html": True, "linkify": True}).enable(["table", "strikethrough", "linkify"])
# Only a URL written with its scheme becomes a link: `memory.md` is a file name, not the address http://memory.md.
assert MD.linkify is not None, "linkify-it-py is not installed (website/requirements.txt)"
MD.linkify.set({"fuzzy_link": False, "fuzzy_email": False})


class Site:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.page_of: dict[str, str] = {}   # repository Markdown file -> site page
        self.anchors: dict[str, str] = {}   # an id in README.md -> the site page that holds it
        self.assets: set[str] = set(COPIED)
        self.missing: list[str] = []        # relative links that point at nothing in the repository
        self.written: set[str] = set()

    def url(self, target: str, page: str, fragment: str = "") -> str:
        rel = posixpath.relpath(target, posixpath.dirname(page) or ".")
        return rel + (f"#{fragment}" if fragment else "")

    def resolve(self, attr: str, url: str, source: str, page: str) -> str:
        """A link as written in `source` (a repository file), rewritten for `page` on the site."""
        if url.startswith("site:"):  # a hand-written page linking to a page of the site
            target, _, fragment = url[5:].partition("#")
            return self.url(target, page, fragment)
        if re.match(r"^([a-z][a-z0-9+.-]*:|//)", url, re.I) or not url:
            return url
        path, _, fragment = url.partition("#")
        if not path:
            if source == "README.md" and self.anchors.get(fragment, page) != page:
                return self.url(self.anchors[fragment], page, fragment)
            return url
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        if target.startswith("..") or not (ROOT / target).exists():
            self.missing.append(f"{source}: {url}")  # broken in the repository too; on the site it is plain text
            return "missing:"
        if target == "README.md":
            return self.url(self.anchors.get(fragment, "index.html"), page, fragment)
        if target in self.page_of:
            return self.url(self.page_of[target], page, fragment)
        if (ROOT / target).is_file() and (attr == "src" or Path(target).suffix.lower() in ASSET):
            self.assets.add(target)
            return self.url(target, page, fragment)
        kind = "tree" if (ROOT / target).is_dir() else "blob"
        return f"{REPO}/{kind}/main/{target}" + (f"#{fragment}" if fragment else "")

    def links(self, body: str, source: str, page: str) -> str:
        def rewrite(m: re.Match) -> str:
            url = self.resolve(m.group(1), html.unescape(m.group(2)), source, page)
            return f'{m.group(1)}="{html.escape(url)}"'

        body = re.sub(rf'<a\b[^>]*\bhref="{re.escape(PRIVATE)}[^"]*"[^>]*>(.*?)</a>',
                      r'<span title="In a private repository">\1</span>', body, flags=re.S)
        body = re.sub(r'\b(href|src)="([^"]*)"', rewrite, body)
        return re.sub(r'<a\b[^>]*\bhref="missing:"[^>]*>(.*?)</a>', r"\1", body, flags=re.S)

    def write(self, page: str, title: str, body: str, *, active: str, wide: bool = False) -> None:
        root = posixpath.relpath(".", posixpath.dirname(page) or ".")
        nav = "".join(f'<a href="{root}/{href}"{CURRENT if href == active else ""}>{label}</a>' for href, label in NAV)
        text = pages.SHELL.format(title=html.escape(title), root=root, nav=nav, body=body, repo=REPO,
                                  logo=pages.LOGO, main="wide" if wide else "doc", public=PUBLIC, page=page)
        path = self.out / page
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.written.add(page)


def slug(text: str) -> str:
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def render(text: str) -> tuple[str, str | None, list[tuple[str, str]]]:
    """Markdown as HTML with GitHub's heading ids; also the first level-1 heading and the level-2 headings."""
    tokens = MD.parse(text)
    seen: dict[str, int] = {}
    title, contents = None, []
    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        words = "".join(c.content for c in tokens[i + 1].children or [] if c.type in ("text", "code_inline"))
        base = slug(words)
        seen[base] = seen.get(base, -1) + 1
        ident = base if not seen[base] else f"{base}-{seen[base]}"
        token.attrSet("id", ident)
        if token.tag == "h1" and title is None:
            title = words
        if token.tag == "h2":
            contents.append((ident, words))
    body = MD.renderer.render(tokens, MD.options, {})
    body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    # A figure gets a box of its own, which scrolls sideways on a phone.
    body = re.sub(r"<p>(<img [^>]*>)</p>", r'<div class="figure">\1</div>', body)
    return body, title, contents


def title_of(source: str, heading: str | None, text: str) -> str:
    """The page title of a document: its level-1 heading, or else its first line if that is short."""
    first = re.sub(r"[*_`#]", "", next((line for line in text.splitlines() if line.strip()), "")).strip()
    return heading or (first if 0 < len(first) <= 80 else source)


def section(text: str, heading: str, level: int = 2) -> str:
    """One section of a Markdown file: from its heading to the next heading of the same or a higher level."""
    lines = text.splitlines()
    start = lines.index("#" * level + " " + heading)
    end = next((i for i in range(start + 1, len(lines))
                if re.match(rf"#{{1,{level}}} ", lines[i]) or (level > 2 and lines[i].startswith("</details>"))),
               len(lines))
    while start and lines[start - 1].startswith("<a id="):  # the anchor GitHub links point at sits above the heading
        start -= 1
    return "\n".join(lines[start:end]) + "\n"


def contents_box(contents: list[tuple[str, str]]) -> str:
    if len(contents) < 3:
        return ""
    items = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in contents)
    return f'<details class="contents"><summary>On this page</summary><ul>{items}</ul></details>'


def build(out: Path = OUT) -> Site:
    if out.exists() and any(out.iterdir()):
        if not (out / ".nojekyll").is_file() or (out / ".git").exists():  # only an earlier build is ever deleted
            raise ValueError(f"{out} is not empty and is not an earlier build of the site; choose another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")  # first, so a build that fails part-way can be run again
    site = Site(out)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for there, here in ON_SITE:
        if there not in readme:
            raise ValueError(f"README.md no longer says {there!r}; update ON_SITE in website/build.py")
        readme = readme.replace(there, here)
    sources = sorted({p.relative_to(ROOT).as_posix() for pattern in MARKDOWN for p in ROOT.glob(pattern)})
    site.page_of = {s: s[:-3] + ".html" for s in sources}

    # README.md first, so that every link to one of its anchors knows which page holds it.
    split = []
    for page, title, headings in README_PAGES:
        body, _, contents = render("".join(section(readme, h) for h in headings))
        for ident in re.findall(r'\bid="([^"]+)"', body):
            site.anchors.setdefault(ident, page)
        split.append((page, title, body, contents))
    for page, title, body, contents in split:
        site.write(page, title, pages.source_line("README.md", REPO) + contents_box(contents)
                   + site.links(body, "README.md", page),
                   active=page if page in dict(NAV) else "how-it-works.html")

    titles: dict[str, str] = {}
    for source in sources:
        page = site.page_of[source]
        text = (ROOT / source).read_text(encoding="utf-8")
        body, heading, contents = render(text)
        titles[source] = title_of(source, heading, text)
        active = "runs.html" if source.startswith("results/") else "docs.html"
        site.write(page, titles[source], pages.source_line(source, REPO) + contents_box(contents)
                   + site.links(body, source, page), active=active)

    data = {"training": demo.training(), "readme": readme, "titles": titles, "repo": REPO}
    for page, title, source, body, wide in pages.written(data, section, render):
        site.write(page, title, site.links(body, source, page), active=page, wide=wide)

    for asset in sorted(site.assets):
        (out / asset).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / asset, out / asset)
    shutil.copytree(ROOT / "website" / "static", out / "static")
    (out / "static" / "favicon.svg").write_text(pages.LOGO, encoding="utf-8")
    return site


def broken(site: Site) -> list[str]:
    """Relative links in the built pages that point at no file of the site."""
    bad = []
    for page in sorted(site.written):
        text = (site.out / page).read_text(encoding="utf-8")
        for url in re.findall(r'\b(?:href|src)="([^"]*)"', text):
            url = html.unescape(url)
            if re.match(r"^([a-z][a-z0-9+.-]*:|//|#)", url, re.I) or not url:
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(page), url.partition("#")[0]))
            if not (site.out / target).is_file():
                bad.append(f"{page}: {url}")
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    site = build(args.out)
    bad = broken(site)
    for line in bad:
        print("broken link:", line)
    for line in site.missing:
        print("points at nothing in the repository:", line)
    print(f"wrote {len(site.written)} pages and {len(site.assets)} files to {os.path.relpath(args.out)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
