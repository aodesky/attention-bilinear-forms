"""Shared project navigation for the generated blueprint pages."""

NAVIGATION = '''<nav class="project-links" aria-label="Project navigation">
  <a href="../">Project home</a>
  <a href="../interactive/gpt2-medium.html">Plots</a>
  <a href="./">Blueprint</a>
  <a href="https://github.com/aodesky/attention-bilinear-forms">GitHub ↗</a>
</nav>
'''


def add_blueprint_navigation(document: str) -> str:
    """Add navigation without changing the renderer's mathematical content."""
    if 'aria-label="Project navigation"' in document:
        return document
    if '</head>' not in document or '</header>' not in document:
        raise ValueError('Blueprint page is missing its head or header')
    document = document.replace('</head>', '<link rel="stylesheet" href="../assets/project-nav.css?v=1">\n</head>', 1)
    return document.replace('</header>', NAVIGATION + '</header>', 1)
