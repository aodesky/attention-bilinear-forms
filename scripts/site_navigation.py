"""Shared top navigation for the project site and generated pages."""

import re


def project_navigation(prefix="../", current=""):
    # Brand icons from https://github.com/simple-icons/simple-icons.
    def icon(name):
        return f'<img src="{prefix}assets/{name}.svg" class="nav-icon" width="14" height="14" alt="" aria-hidden="true">'

    links = (
        ("home", prefix, "Attention &amp; bilinear forms"),
        ("plots", prefix + "interactive/gpt2-medium.html", "Plots"),
        ("blueprint", prefix + "blueprint/", "Blueprint"),
        ("github", "https://github.com/aodesky/attention-bilinear-forms", f'{icon("github")}GitHub'),
    )
    items = []
    for key, href, label in links:
        attributes = ' class="project-name"' if key == "home" else ""
        if key == current:
            attributes += ' aria-current="page"'
        items.append(f'  <a href="{href}"{attributes}>{label}</a>')
    # Replace this placeholder with the paper link once it is on arXiv.
    items.append(f'  <span class="paper-forthcoming">{icon("arxiv")}arXiv (forthcoming)</span>')
    items.append('  <a href="https://andrewodesky.com">Andrew O’Desky</a>')
    return '<nav class="project-links" aria-label="Project navigation">\n' + "\n".join(items) + "\n</nav>\n"


def add_blueprint_navigation(document: str) -> str:
    """Install the top bar while preserving the renderer's own header and content."""
    if '</head>' not in document or '<body>' not in document:
        raise ValueError('Blueprint page is missing its head or body')
    document = re.sub(r'<nav class="project-links".*?</nav>\n?', '', document, flags=re.S)
    document = re.sub(r'<link rel="stylesheet" href="../assets/project-nav.css[^"]*">\n?', '', document)
    document = document.replace('</head>', '<link rel="stylesheet" href="../assets/project-nav.css?v=5">\n</head>', 1)
    return document.replace('<body>\n', '<body>\n' + project_navigation(current="blueprint"), 1)
