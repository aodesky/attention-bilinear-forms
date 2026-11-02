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
    updateControls();
    decorate();
  }

  function ensureFavicon() {
  if (!document.querySelector('link[rel~="icon"]')) {
    [['icon', 'image/svg+xml', 'favicon.svg'], ['icon', 'image/png', 'favicon-32x32.png'], ['apple-touch-icon', 'image/png', 'apple-touch-icon.png']].forEach(function (item) {
      var link = document.createElement('link');
      link.rel = item[0]; link.type = item[1];
      link.href = 'https://andrewodesky.com/res/favicon/' + item[2];
      if (item[2] === 'favicon-32x32.png') link.sizes = '32x32';
      document.head.appendChild(link);
    });
  }
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

  var ICONS = {
    system: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    light: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    dark: '<path d="M20.9 13.1A9 9 0 0 1 10.9 3.1 9 9 0 1 0 20.9 13.1Z"/>'
  };
  var LABELS = {system: 'System', light: 'Light', dark: 'Dark'};
  function icon(body, className) {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"' + (className ? ' class="' + className + '"' : '') + '>' + body + '</svg>';
  }
  function updateControls() {
    if (!ICONS) return;
    var pickers = document.querySelectorAll('.theme-picker');
    for (var i = 0; i < pickers.length; i++) {
      var trigger = pickers[i].querySelector('.theme-toggle');
      if (!trigger) continue;
      trigger.innerHTML = icon(ICONS[preference]) + icon('<path d="m6 9 6 6 6-6"/>', 'theme-chevron');
      trigger.setAttribute('aria-label', 'Color theme: ' + LABELS[preference]);
      trigger.title = 'Color theme: ' + LABELS[preference];
      var options = pickers[i].querySelectorAll('[data-theme-choice]');
      for (var j = 0; j < options.length; j++) options[j].setAttribute('aria-checked', String(options[j].dataset.themeChoice === preference));
    }
  }
  function close(picker, focus) {
    picker.querySelector('.theme-menu').hidden = true;
    var trigger = picker.querySelector('.theme-toggle');
    trigger.setAttribute('aria-expanded', 'false');
    if (focus) trigger.focus();
  }
  function wire() {
    ensureFavicon();
    var pickers = document.querySelectorAll('.theme-picker');
    for (var i = 0; i < pickers.length; i++) {
      (function (picker) {
        picker.innerHTML = '<button type="button" class="theme-toggle" aria-haspopup="menu" aria-expanded="false"></button><div class="theme-menu" role="menu" aria-label="Color theme" hidden></div>';
        var trigger = picker.querySelector('.theme-toggle');
        var menu = picker.querySelector('.theme-menu');
        ['system', 'light', 'dark'].forEach(function (choice) {
          var option = document.createElement('button');
          option.type = 'button';
          option.className = 'theme-option';
          option.dataset.themeChoice = choice;
          option.setAttribute('role', 'menuitemradio');
          option.setAttribute('aria-label', LABELS[choice]);
          option.title = LABELS[choice];
          option.tabIndex = -1;
          option.innerHTML = icon(ICONS[choice]) + icon('<path d="m5 12 4 4L19 6"/>', 'theme-check');
          option.addEventListener('click', function () { save(choice); apply(choice); close(picker, true); });
          menu.appendChild(option);
        });
        function open(index) {
          menu.hidden = false;
          trigger.setAttribute('aria-expanded', 'true');
          (index == null ? menu.querySelector('[aria-checked="true"]') : menu.children[index]).focus();
        }
        trigger.addEventListener('click', function () { if (menu.hidden) open(); else close(picker, false); });
        trigger.addEventListener('keydown', function (event) {
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault(); open(event.key === 'ArrowDown' ? 0 : 2);
          }
        });
        menu.addEventListener('keydown', function (event) {
          var index = Array.prototype.indexOf.call(menu.children, document.activeElement);
          if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(picker, true); }
          else if (event.key === 'Tab') close(picker, false);
          else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].indexOf(event.key) !== -1) {
            event.preventDefault();
            var next = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (index + (event.key === 'ArrowDown' ? 1 : 2)) % 3;
            menu.children[next].focus();
          }
        });
      })(pickers[i]);
    }
    document.addEventListener('pointerdown', function (event) {
      for (var i = 0; i < pickers.length; i++) if (!pickers[i].contains(event.target)) close(pickers[i], false);
    });
    updateControls();
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
