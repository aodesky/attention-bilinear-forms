'use strict';

const figures = [
  { key: 'simplex', title: 'Interactive 3-D simplex', file: 'simplex.html', help: 'Drag to rotate, scroll to zoom, and hover for layer and head details.' },
  { key: 'delta', title: '2-D projection to Δ', file: 'delta.png', help: 'Projection of the head profiles onto the face of symmetric profiles.' },
  { key: 'extreme-histograms', title: 'Spectral distributions by parity (k = 3)', file: 'extreme-histograms.png', help: 'Mean histograms of the extreme members of each spectral type.' },
  { key: 'clusters-by-layer', title: 'Clusters by layer (k = 3)', file: 'clusters-by-layer.png', help: 'The distribution of the three spectral types across model layers.' },
];
const model = document.getElementById('model');
let currentFigure = 0;
const frame = document.getElementById('interactive-figure');
const picture = document.getElementById('static-figure');
const error = document.getElementById('figure-error');
const buttons = Array.from(document.querySelectorAll('[data-figure]'));

function showFigure(index, updateURL = true) {
  const figure = figures[index] || figures[0];
  index = figures.indexOf(figure);
  currentFigure = index;
  document.getElementById('figure-title').textContent = figure.title;
  document.getElementById('figure-help').textContent = figure.help;
  document.getElementById('viewer').setAttribute('aria-labelledby', `figure-tab-${index}`);
  document.title = `${model.value} · ${figure.title}`;
  const source = `assets/${model.value}/${figure.file}?v=profile-titles-2`;
  document.getElementById('open-figure').href = source;
  error.hidden = true;
  frame.hidden = index !== 0;
  picture.hidden = index === 0;
  if (index === 0) {
    // Keep the existing frame alive so its camera and RoPE position survive.
    if (frame.getAttribute('src') !== source) frame.src = source;
    frame.title = `${figure.title} for ${model.value}`;
    frame.contentWindow?.dispatchEvent(new Event('resize'));
  } else {
    picture.alt = `${figure.title} for ${model.value}`;
    picture.src = source;
  }
  for (const button of buttons) {
    const selected = Number(button.dataset.figure) === index;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  if (updateURL) {
    const url = new URL(location.href);
    url.hash = figure.key;
    history.replaceState(null, '', url);
  }
}

function showHash() {
  const index = figures.findIndex(figure => `#${figure.key}` === location.hash);
  showFigure(Math.max(0, index), false);
}

for (const button of buttons) {
  button.addEventListener('click', () => showFigure(Number(button.dataset.figure)));
  button.addEventListener('keydown', event => {
    let next = currentFigure;
    if (event.key === 'ArrowRight') next = (currentFigure + 1) % figures.length;
    else if (event.key === 'ArrowLeft') next = (currentFigure + figures.length - 1) % figures.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = figures.length - 1;
    else return;
    event.preventDefault();
    showFigure(next);
    buttons[next].focus();
  });
}
model.addEventListener('change', () => {
  location.assign(`${model.value}.html#${figures[currentFigure].key}`);
});
picture.addEventListener('error', () => { error.hidden = false; });
picture.addEventListener('load', () => { error.hidden = true; });
window.addEventListener('hashchange', showHash);
showHash();
