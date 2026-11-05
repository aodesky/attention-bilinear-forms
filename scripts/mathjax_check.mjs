// Typeset every math span of one HTML page with MathJax and report errors.
//
// Input (JSON on stdin): {"html": page markup, "component": "tex-chtml",
//   "configScripts": [inline <script> sources that set the MathJax config]}.
// Output (JSON on stdout): {"expressions": n, "errors": [{kind, message, snippet}]}.
// The page's own configuration is used, with two changes: TeX errors are
// thrown instead of rendered, and the noundefined/noerrors packages are
// removed so undefined macros are errors. A fresh process is needed per page
// because \newcommand definitions persist in the TeX input jax.

import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

// MathJax's minified source makes Node's default crash report unreadable.
for (const event of ['uncaughtException', 'unhandledRejection']) {
  process.on(event, (error) => {
    console.error(error?.stack?.split('\n').slice(0, 4).map((l) => l.slice(0, 300)).join('\n')
                  ?? String(error));
    process.exit(2);
  });
}

const require = createRequire(import.meta.url);
const input = JSON.parse(readFileSync(0, 'utf8'));
const html = input.html;

function pageConfig(scripts) {
  const context = {};
  context.window = context;
  vm.createContext(context);
  for (const source of scripts) vm.runInContext(source, context, {timeout: 1000});
  if (scripts.length && context.MathJax === undefined) {
    throw new Error('MathJax configuration scripts did not define MathJax');
  }
  // Round-trip through JSON to leave the sandbox; functions are dropped.
  return JSON.parse(JSON.stringify(context.MathJax ?? {}));
}

function withoutPackages(packages, removed) {
  if (Array.isArray(packages)) return packages.filter((p) => !removed.includes(p));
  const result = {...(packages ?? {})};
  result['[+]'] = (result['[+]'] ?? []).filter((p) => !removed.includes(p));
  result['[-]'] = [...(result['[-]'] ?? []), ...removed];
  return result;
}

const component = input.component;
const config = pageConfig(input.configScripts);
const hasTeX = /^tex-/.test(component);
const load = [...(config.loader?.load ?? []), 'adaptors/liteDOM'];
if (hasTeX) load.unshift(/-full(-|$)/.test(component) ? 'input/tex-full' : 'input/tex');
config.loader = {...(config.loader ?? {}), load, versionWarnings: false};
config.startup = {typeset: false};
if (hasTeX) {
  config.tex = {
    ...(config.tex ?? {}),
    packages: withoutPackages(config.tex?.packages, ['noundefined', 'noerrors']),
    formatError: (jax, error) => { throw error; },
  };
}

const MathJax = await require('mathjax').init(config);
const {mathjax} = MathJax._.mathjax;
const adaptor = MathJax.startup.adaptor;
const doc = MathJax.startup.getDocument(html);
const errors = [];
const shorten = (s) => (s.length > 200 ? s.slice(0, 200) + '…' : s).replace(/\s+/g, ' ');
let expressions = 0;

if (hasTeX) {
  doc.findMath();
  for (const item of doc.math) {
    if (item.display === null) continue; // an escaped delimiter such as \$
    expressions += 1;
    const delimited = (item.start.delim ?? '') + item.math + (item.end.delim ?? '');
    if (item.start.node !== item.end.node) {
      errors.push({kind: 'math crosses HTML elements',
                   message: 'a delimiter is probably unmatched', snippet: shorten(delimited)});
    }
    try {
      await mathjax.handleRetriesFor(() => item.compile(doc));
    } catch (error) {
      errors.push({kind: error.id ? `TeX ${error.id}` : 'MathJax error',
                   message: error.message, snippet: shorten(delimited)});
    }
  }
  // Replace each math span by a placeholder so the remaining text is what a
  // reader sees as prose.
  for (const item of [...doc.math].reverse()) {
    item.typesetRoot = adaptor.node('mjx-container');
    item.updateDocument(doc);
  }
}

// Search the visible prose (and the page title) for TeX that was not converted.
const skipped = new Set(['script', 'style', 'noscript', 'textarea', 'pre', 'code',
                         'annotation', 'annotation-xml', 'mjx-container', 'svg']);
// One report per text node, for the first kind of leftover TeX found.
const RAW = [
  [/\\[()[\]]|\\begin\{/g, 'math left as text (unmatched delimiter or unbalanced braces)'],
  [/\$/g, 'stray $'],
  [/\\(?:[A-Za-z]+|[{}_^&%#\\])/g, 'TeX command outside math'],
];
// Text that MathJax is told to ignore may contain TeX on purpose.
const ignored = new RegExp(`(^| )(?:${config.options?.ignoreHtmlClass ?? 'tex2jax_ignore'})( |$)`);
function scan(node) {
  const kind = adaptor.kind(node);
  if (kind === '#text') {
    const text = adaptor.value(node);
    for (const [pattern, label] of RAW) {
      const matches = [...text.matchAll(pattern)];
      if (!matches.length) continue;
      const first = matches[0].index;
      errors.push({kind: label, message: [...new Set(matches.map((m) => m[0]))].join(' '),
                   snippet: shorten(text.slice(Math.max(0, first - 60), first + 100))});
      break;
    }
  } else if (!kind.startsWith('#') && !skipped.has(kind)
             && !ignored.test(adaptor.getAttribute(node, 'class') ?? '')) {
    for (const child of adaptor.childNodes(node)) scan(child);
  }
}
const root = adaptor.root(doc.document);
for (const title of adaptor.tags(root, 'title')) scan(title);
scan(adaptor.body(doc.document));

process.stdout.write(JSON.stringify({expressions, errors}));
