#!/usr/bin/env python3
"""Typeset every math span of a built site with MathJax and report errors.

Checks SITE/index.html and SITE/blueprint/*.html. Each page is typeset with
the MathJax version pinned in package.json (which must match the version the
page loads) and with the page's own MathJax configuration. The check fails on
TeX parse errors, undefined macros or environments, unbalanced braces, and on
TeX left in the visible text: stray $, \\( \\) \\[ \\], or commands such as
\\cite that the TeX-to-HTML converter did not handle.

Setup: run `npm ci` in the directory containing package.json (the parent of
this scripts directory).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).resolve().with_name('mathjax_check.mjs')
MATHJAX_PACKAGE = ROOT / 'node_modules/mathjax/package.json'
# A MathJax combined component, e.g. .../mathjax@3/es5/tex-chtml.js.
MATHJAX_SRC = re.compile(r'mathjax(?:@|/)(?P<version>[^/]*)/es5/(?P<component>[\w-]+)\.js')
# Script types a browser executes.
JAVASCRIPT = {'', 'text/javascript', 'application/javascript', 'module'}


class Page(HTMLParser):
    """Collect a page's scripts and re-serialize it with quoted attributes.

    MathJax's node DOM parser mishandles unquoted attribute values such as
    <nav class=local_toc>, which browsers accept; quoting them is the only
    change made to the markup.
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.sources = []   # (src, inline text) of executed scripts, in order
        self.out = []
        self._inline = None

    def handle_starttag(self, tag, attrs):
        self.out.append(self._tag(tag, attrs, ''))
        if tag == 'script':
            attrs = dict(attrs)
            src = attrs.get('src')
            if (attrs.get('type') or 'text/javascript').lower() not in JAVASCRIPT:
                # Browsers do not run e.g. MathJax 2's text/x-mathjax-config.
                self._inline = None
            elif src is None:
                self._inline = []
            else:
                self.sources.append((src, None))

    def handle_startendtag(self, tag, attrs):
        self.out.append(self._tag(tag, attrs, ' /'))

    def handle_data(self, data):
        self.out.append(data)
        if self._inline is not None:
            self._inline.append(data)

    def handle_endtag(self, tag):
        self.out.append(f'</{tag}>')
        if tag == 'script' and self._inline is not None:
            self.sources.append((None, ''.join(self._inline)))
            self._inline = None

    def handle_entityref(self, name):
        self.out.append(f'&{name};')

    def handle_charref(self, name):
        self.out.append(f'&#{name};')

    def handle_comment(self, data):
        self.out.append(f'<!--{data}-->')

    def handle_decl(self, decl):
        self.out.append(f'<!{decl}>')

    def unknown_decl(self, data):
        self.out.append(f'<![{data}]>')

    def handle_pi(self, data):
        self.out.append(f'<?{data}>')

    @staticmethod
    def _tag(tag, attrs, close):
        attributes = ''.join(f' {k}' if v is None else f' {k}="{escape(v)}"'
                             for k, v in attrs)
        return f'<{tag}{attributes}{close}>'


def page_setup(path, installed):
    """Return the page markup, MathJax component and config scripts."""
    parser = Page()
    parser.feed(path.read_text())
    parser.close()
    html = ''.join(parser.out)
    loaders = [i for i, (src, _) in enumerate(parser.sources)
               if src and re.search('mathjax', src, re.I)]
    if not loaders:
        # A page without MathJax still must not show raw TeX.
        return html, 'none', []
    if len(loaders) > 1:
        raise ValueError(f'{path}: more than one MathJax script')
    src = parser.sources[loaders[0]][0]
    match = MATHJAX_SRC.search(src)
    if not match:
        raise ValueError(f'{path}: unrecognized MathJax script {src}')
    version = match['version']
    if not re.fullmatch(r'\d+(\.\d+){0,2}', version):
        raise ValueError(f'{path}: MathJax script {src} does not name a version')
    if installed.split('.')[:version.count('.') + 1] != version.split('.'):
        raise ValueError(f'{path}: page loads MathJax {version} but {installed} '
                         f'is installed; update package.json to match')
    configs = [text for src, text in parser.sources[:loaders[0]]
               if text is not None and re.search(r'\bMathJax\b', text)]
    return html, match['component'], configs


def check_page(node, path, setup):
    html, component, configs = setup
    request = json.dumps({'html': html, 'component': component, 'configScripts': configs})
    result = subprocess.run([node, str(HELPER)], input=request, capture_output=True,
                            text=True)
    if result.returncode:
        raise RuntimeError(f'MathJax check crashed on {path}:\n{result.stderr}')
    return json.loads(result.stdout)


def site_pages(site):
    pages = sorted((site / 'blueprint').glob('*.html'))
    if (site / 'index.html').is_file():
        pages.insert(0, site / 'index.html')
    if not pages:
        raise ValueError(f'No HTML pages in {site}')
    return pages


def check(site):
    """Return (errors, expression count, page count) for the site directory."""
    node = shutil.which('node')
    if node is None:
        raise RuntimeError('node is required for the math check (see README)')
    if not MATHJAX_PACKAGE.is_file():
        raise RuntimeError(f'MathJax is not installed: run `npm ci` in {ROOT}')
    installed = json.loads(MATHJAX_PACKAGE.read_text())['version']
    pages = site_pages(site)
    setups = {path: page_setup(path, installed) for path in pages}
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        results = dict(zip(pages, pool.map(lambda p: check_page(node, p, setups[p]),
                                           pages)))
    errors = [f'{path.relative_to(site)}: {e["kind"]}: {e["message"]}\n    {e["snippet"]}'
              for path, result in results.items() for e in result['errors']]
    return errors, sum(r['expressions'] for r in results.values()), len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('site', nargs='?', type=Path, default=ROOT,
                        help='site directory (default: %(default)s)')
    site = parser.parse_args().site.resolve()
    errors, expressions, pages = check(site)
    if errors:
        raise SystemExit('\n'.join(errors) + f'\n{len(errors)} math rendering problems')
    print(f'Typeset {expressions} math expressions on {pages} pages without errors.')


if __name__ == '__main__':
    main()
