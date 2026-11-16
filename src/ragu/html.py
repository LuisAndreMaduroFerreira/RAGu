from bs4 import BeautifulSoup, Tag


def _clean(soup: Tag) -> str:
    for element in reversed(soup.find_all(["script", "style", "nav"])):
        element.decompose()
    for heading in reversed(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])):
        heading.replace_with(
            f"\n{'#' * int(heading.name[1])} {heading.get_text(' ', strip=True)}\n"
        )
    for pre in reversed(soup.find_all("pre")):
        pre.replace_with(f"\n```\n{pre.get_text()}\n```\n")
    for row in reversed(soup.find_all("tr")):
        row.replace_with(
            "\n"
            + " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"]))
            + "\n"
        )
    return soup.get_text("\n", strip=True)


def clean_html(html: str) -> str:
    return _clean(BeautifulSoup(html, "html.parser"))


def read_html_document(html: str) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    for element in reversed(
        soup.find_all(["head", "template", "noscript", "iframe", "object", "embed"])
    ):
        element.decompose()
    for element in reversed(soup.select("[hidden], [aria-hidden='true' i]")):
        element.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    return title, _clean(body)
