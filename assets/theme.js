/* Shared theme handling for andrewodesky.com and the project pages at
   aodesky.github.io.  Identical copy in each repository.

   - The stored preference is "dark" or "light"; absent means follow the OS.
   - A ?theme= parameter wins over storage on arrival, so a link followed from
     a sibling site opens in the mode the visitor was already reading in. The
     parameter is stripped from the address bar afterwards.
   - Outbound links to the sibling sites carry the current mode, since the two
     origins cannot share storage.

   Run this in <head>, before the page renders, to avoid a flash of the wrong
   theme. It touches only documentElement, which exists at that point. */
(function () {
  var KEY = 'theme';
  // Hosts whose links should carry the current mode.
  var SIBLINGS = /(^|\.)andrewodesky\.com$|(^|\.)aodesky\.github\.io$/;

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }

  function save(mode) {
    try { localStorage.setItem(KEY, mode); } catch (e) { /* private mode */ }
  }

  function systemDark() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  /* The mode actually being displayed, whether chosen or inherited from the OS. */
  function current() {
    var explicit = document.documentElement.getAttribute('data-theme');
    if (explicit) return explicit;
    return systemDark() ? 'dark' : 'light';
  }

  function apply(mode) {
    if (mode === 'dark' || mode === 'light') {
      document.documentElement.setAttribute('data-theme', mode);
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }

  /* On arrival: a ?theme= parameter from a sibling site beats stored state. */
  var incoming = null;
  try {
    incoming = new URLSearchParams(window.location.search).get(KEY);
  } catch (e) { /* very old browser */ }

  if (incoming === 'dark' || incoming === 'light') {
    apply(incoming);
    save(incoming);
    // Drop the parameter so it is not carried into bookmarks or shares.
    try {
      var url = new URL(window.location.href);
      url.searchParams.delete(KEY);
      history.replaceState(null, '', url.pathname + url.search + url.hash);
    } catch (e) { /* replaceState unavailable */ }
  } else {
    apply(stored());
  }

  /* Expose a small API for the toggle button, wired up on DOMContentLoaded. */
  window.themeToggle = function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    apply(next);
    save(next);
    decorate();
  };

  /* Give sibling-site links the current mode. Re-run after a toggle so links
     reflect the latest choice. */
  function decorate() {
    var mode = current();
    var links = document.querySelectorAll('a[href]');
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var url;
      try { url = new URL(a.href, window.location.href); } catch (e) { continue; }
      if (url.origin === window.location.origin) continue;
      if (!SIBLINGS.test(url.hostname)) continue;
      url.searchParams.set(KEY, mode);
      a.href = url.toString();
    }
  }

  function wire() {
    var btn = document.querySelector('.theme-toggle');
    if (btn) {
      btn.addEventListener('click', window.themeToggle);
      btn.setAttribute('aria-label', 'Switch between dark and light mode');
    }
    decorate();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }

  /* Follow the OS while the visitor has expressed no preference. */
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onChange = function () { if (!stored()) decorate(); };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }
})();
