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
    items.append(theme_selector())
    return '<nav class="project-links" aria-label="Project navigation">\n' + "\n".join(items) + "\n</nav>\n"


def theme_selector():
    """The shared script supplies the same icon menu on every site."""
    return '<div class="theme-picker"></div>'


def theme_script_tag(prefix="../"):
    """The shared theme script, loaded in <head> so there is no flash."""
    return f'<script src="{prefix}assets/theme.js?v=polish-1"></script>'


# Set the canvas color before stylesheets or MathJax can delay the first paint.
EARLY_THEME = """<style id="early-theme-style">html { background: #fff; } html[data-theme="dark"] { background: #14171a; }</style>
<script id="early-theme">(function () {
  var choice = null;
  try { choice = new URLSearchParams(location.search).get('theme'); } catch (_) {}
  if (choice !== 'system' && choice !== 'light' && choice !== 'dark') {
    try { choice = localStorage.getItem('theme'); } catch (_) {}
  }
  if (choice !== 'light' && choice !== 'dark') choice = 'system';
  var mode = choice === 'system' ? (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : choice;
  document.documentElement.setAttribute('data-theme', mode);
  document.documentElement.setAttribute('data-theme-preference', choice);
  document.documentElement.style.colorScheme = mode;
})();</script>
"""


def add_blueprint_navigation(document: str) -> str:
    """Install the top bar while preserving the renderer's own header and content."""
    if '</head>' not in document or '<body>' not in document:
        raise ValueError('Blueprint page is missing its head or body')
    document = re.sub(r'<style id="early-theme-style">.*?</style>\s*<script id="early-theme">.*?</script>\s*', '', document, flags=re.S)
    document = document.replace('<head>', '<head>\n' + EARLY_THEME, 1)
    document = re.sub(r'<nav class="project-links".*?</nav>\n?', '', document, flags=re.S)
    document = re.sub(r'<link rel="stylesheet" href="../assets/project-nav.css[^"]*">\n?', '', document)
    document = re.sub(r'<script src="../assets/theme.js[^"]*"></script>\n?', '', document)
    document = re.sub(r'<link[^>]+href="https://andrewodesky.com/res/favicon/[^"]+"[^>]*>\n?', '', document)
    document = re.sub(r'<link[^>]+href="../assets/theme.css[^"]*"[^>]*>\n?', '', document)
    document = re.sub(r'<script[^>]+src="../assets/blueprint.js[^"]*"[^>]*></script>\n?', '', document)
    document = document.replace(
        '</head>',
        '<link rel="stylesheet" href="../assets/project-nav.css?v=polish-1">\n'
        + '<link rel="stylesheet" href="../assets/theme.css?v=polish-1">\n'
        + '<script src="../assets/blueprint.js?v=polish-1" defer></script>\n'
        + '<link rel="icon" type="image/svg+xml" href="https://andrewodesky.com/res/favicon/favicon.svg">\n<link rel="icon" type="image/png" sizes="32x32" href="https://andrewodesky.com/res/favicon/favicon-32x32.png">\n<link rel="apple-touch-icon" href="https://andrewodesky.com/res/favicon/apple-touch-icon.png">\n'
        + theme_script_tag('../') + '\n</head>',
        1,
    )
    return document.replace('<body>\n', '<body>\n' + project_navigation(current="blueprint"), 1)
