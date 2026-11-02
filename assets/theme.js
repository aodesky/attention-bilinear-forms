/* Shared theme preference for andrewodesky.com and aodesky.github.io.
   System is the default. Links between sites carry the preference itself,
   so following a link never turns System into a fixed light/dark choice.
   Load in <head> before rendering. */
(function () {
  var KEY = 'theme';
  var SIBLINGS = /(^|\.)andrewodesky\.com$|(^|\.)aodesky\.github\.io$/;
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var preference = 'system';

  function valid(value) {
    return value === 'system' || value === 'light' || value === 'dark';
  }

  function stored() {
    try {
      var value = localStorage.getItem(KEY);
      return valid(value) ? value : 'system';
    } catch (e) { return preference; }
  }

  function save(value) {
    try { localStorage.setItem(KEY, value); } catch (e) { /* storage unavailable */ }
  }

  function decorate() {
    var links = document.querySelectorAll('a[href]');
    for (var i = 0; i < links.length; i++) {
      var url;
      try { url = new URL(links[i].href, location.href); } catch (e) { continue; }
      if (!/^https?:$/.test(url.protocol) || !SIBLINGS.test(url.hostname)) continue;
      var otherProject = url.hostname === 'aodesky.github.io' &&
        url.pathname.split('/')[1] !== location.pathname.split('/')[1];
      if (url.origin === location.origin && !otherProject) continue;
      url.searchParams.set(KEY, preference);
      links[i].href = url.toString();
    }
  }

  function apply(value) {
    preference = valid(value) ? value : 'system';
    var mode = preference === 'system' ? (media && media.matches ? 'dark' : 'light') : preference;
    root.setAttribute('data-theme', mode);
    root.setAttribute('data-theme-preference', preference);
    root.style.colorScheme = mode;
    var controls = document.querySelectorAll('.theme-select');
    for (var i = 0; i < controls.length; i++) controls[i].value = preference;
    decorate();
  }

  var incoming = null;
  try { incoming = new URLSearchParams(location.search).get(KEY); } catch (e) {}
  apply(valid(incoming) ? incoming : stored());
  if (valid(incoming)) {
    save(incoming);
    try {
      var url = new URL(location.href);
      url.searchParams.delete(KEY);
      history.replaceState(history.state, '', url.pathname + url.search + url.hash);
    } catch (e) { /* history unavailable */ }
  }

  function wire() {
    var controls = document.querySelectorAll('.theme-select');
    for (var i = 0; i < controls.length; i++) {
      controls[i].value = preference;
      controls[i].addEventListener('change', function () {
        save(this.value);
        apply(this.value);
      });
    }
    decorate();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire);
  else wire();

  if (media) {
    var onChange = function () { if (preference === 'system') apply('system'); };
    if (media.addEventListener) media.addEventListener('change', onChange);
    else if (media.addListener) media.addListener(onChange);
  }
  window.addEventListener('storage', function (event) {
    if (event.key === KEY || event.key === null) apply(stored());
  });
  window.addEventListener('pageshow', function (event) {
    if (event.persisted) apply(stored());
  });
})();
