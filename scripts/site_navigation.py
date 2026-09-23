"""Shared top navigation for the project site and generated pages."""

import re


def project_navigation(prefix="../", current=""):
    # Brand icons from https://github.com/simple-icons/simple-icons.
    def icon(name):
        return f'<img src="{prefix}assets/{name}.svg" class="nav-icon" width="14" height="14" alt="" aria-hidden="true">'

    links = (
        ("home", prefix, 'Attention<span class="nav-long"> &amp; bilinear forms</span>'),
        ("plots", prefix + "interactive/gpt2-medium.html", "Plots"),
        ("blueprint", prefix + "blueprint/", "Blueprint"),
        ("github", "https://github.com/aodesky/attention-bilinear-forms",
         f'{icon("github")}<span class="nav-label">GitHub</span>'),
        ("arxiv", "https://arxiv.org/abs/2609.22990",
         f'{icon("arxiv")}<span class="nav-label">arXiv</span>'),
    )
    items = []
    for key, href, label in links:
        attributes = ' class="project-name"' if key == "home" else ""
        if key in ("github", "arxiv"):
            label_text = "GitHub" if key == "github" else "arXiv"
            attributes += f' aria-label="{label_text}" title="{label_text}"'  # the text label is hidden on narrow screens
        if key == current:
            attributes += ' aria-current="page"'
        items.append(f'  <a href="{href}"{attributes}>{label}</a>')
    items.append('  <a href="https://andrewodesky.com" class="author-link">Andrew O’Desky</a>')
    items.append(theme_toggle_button(prefix))
    return '<nav class="project-links" aria-label="Project navigation">\n' + "\n".join(items) + "\n</nav>\n"


MOON_PATH = "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"
SUN_PATH = (
    "M12 17a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm0-13a1 1 0 0 1 1 1v1a1 1 0 1 1-2 0V5a1 1 0 0 1 1-1zm0 14a1 1 0 0 1 1 1v1a1 1 0 1 1-2 0v-1a1 1 0 0 1 1-1zM4 12a1 1 0 0 1 1-1h1a1 1 0 1 1 0 2H5a1 1 0 0 1-1-1zm14 0a1 1 0 0 1 1-1h1a1 1 0 1 1 0 2h-1a1 1 0 0 1-1-1zM6.3 6.3a1 1 0 0 1 1.4 0l.7.7a1 1 0 0 1-1.4 1.4l-.7-.7a1 1 0 0 1 0-1.4zm9.3 9.3a1 1 0 0 1 1.4 0l.7.7a1 1 0 0 1-1.4 1.4l-.7-.7a1 1 0 0 1 0-1.4zm2.1-9.3a1 1 0 0 1 0 1.4l-.7.7a1 1 0 1 1-1.4-1.4l.7-.7a1 1 0 0 1 1.4 0zM8.4 15.6a1 1 0 0 1 0 1.4l-.7.7a1 1 0 0 1-1.4-1.4l.7-.7a1 1 0 0 1 1.4 0z"
)


def theme_toggle_button(prefix="../"):
    """The dark/light toggle, last in the bar; same markup as andrewodesky.com."""
    return (
        '  <button type="button" class="theme-toggle"'
        ' title="Switch between dark and light mode">'
        f'<svg class="icon-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="{MOON_PATH}"/></svg>'
        f'<svg class="icon-sun" viewBox="0 0 24 24" aria-hidden="true"><path d="{SUN_PATH}"/></svg>'
        '</button>'
    )


def theme_script_tag(prefix="../"):
    """The shared theme script, loaded in <head> so there is no flash."""
    return f'<script src="{prefix}assets/theme.js"></script>'


def add_blueprint_navigation(document: str) -> str:
    """Install the top bar while preserving the renderer's own header and content."""
    if '</head>' not in document or '<body>' not in document:
        raise ValueError('Blueprint page is missing its head or body')
    document = re.sub(r'<nav class="project-links".*?</nav>\n?', '', document, flags=re.S)
    document = re.sub(r'<link rel="stylesheet" href="../assets/project-nav.css[^"]*">\n?', '', document)
    document = re.sub(r'<script src="../assets/theme.js"></script>\n?', '', document)
    document = document.replace(
        '</head>',
        '<link rel="stylesheet" href="../assets/project-nav.css?v=7">\n'
        + theme_script_tag('../') + '\n</head>',
        1,
    )
    return document.replace('<body>\n', '<body>\n' + project_navigation(current="blueprint"), 1)
