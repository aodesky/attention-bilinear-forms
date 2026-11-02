/* Shared navigation controls for generated blueprint pages. */
(function () {
  function setup() {
    if (document.body.classList.contains('blueprint-ready')) return;
    var nav = document.querySelector('.project-links');
    var toc = document.querySelector('nav.toc');
    if (!nav || !toc) return;
    document.body.classList.add('blueprint-ready');
    toc.id = 'blueprint-contents';
    toc.setAttribute('aria-label', 'Contents');
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'sidebar-toggle';
    button.setAttribute('aria-controls', toc.id);
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/></svg><span>Contents</span>';
    var brand = nav.querySelector('.project-name');
    brand.insertAdjacentElement('afterend', button);
    var narrow = window.matchMedia('(max-width: 900px)');
    var preference = null;
    try { preference = localStorage.getItem('blueprint-contents'); } catch (e) {}
    var backdrop = document.createElement('button');
    backdrop.type = 'button';
    backdrop.className = 'sidebar-backdrop';
    backdrop.setAttribute('aria-label', 'Close contents');
    backdrop.tabIndex = -1;
    document.querySelector('.wrapper').appendChild(backdrop);
    function show(open, focus) {
      toc.hidden = !open;
      backdrop.hidden = !open || !narrow.matches;
      button.setAttribute('aria-expanded', String(open));
      button.setAttribute('aria-label', open ? 'Hide contents' : 'Show contents');
      button.title = open ? 'Hide contents' : 'Show contents';
      if (focus) (open && narrow.matches ? toc.querySelector('a') : button).focus();
    }
    function initial() { show(!narrow.matches && preference !== 'closed', false); }
    button.addEventListener('click', function () {
      var open = toc.hidden;
      if (!narrow.matches) {
        preference = open ? 'open' : 'closed';
        try { localStorage.setItem('blueprint-contents', preference); } catch (e) {}
      }
      show(open, true);
    });
    backdrop.addEventListener('click', function () { show(false, true); });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && narrow.matches && !toc.hidden) show(false, true);
    });
    toc.addEventListener('click', function (event) {
      if (narrow.matches && event.target.closest('a')) show(false, false);
    });
    if (narrow.addEventListener) narrow.addEventListener('change', initial);
    else narrow.addListener(initial);
    initial();
  }
  document.addEventListener('DOMContentLoaded', setup, {once: true});
  if (document.readyState !== 'loading') setup();
})();
