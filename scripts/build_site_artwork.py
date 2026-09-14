#!/usr/bin/env python3
"""Generate the site's illustration and social card from actual GPT-2 profiles.

Run with Python and Pillow installed. The SVG requires only the standard
library; the PNG uses Georgia when available, with a DejaVu fallback.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERTICES = [(245, 294), (68, 176), (373, 202), (240, 40)]


def data():
    with (ROOT / 'scripts/survey_results/head_statistics.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['model'] == 'gpt2']
    assert len(rows) == 144
    points = []
    for r in rows:
        w = [float(r[k]) for k in 'abcd']
        assert abs(sum(w) - 1) < 1e-10
        points.append(tuple(sum(w[i] * VERTICES[i][j] for i in range(4)) for j in range(2)))
    return points


def main():
    points = data()
    output = ROOT / 'assets'
    output.mkdir(exist_ok=True)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 340" role="img" aria-labelledby="title desc">',
           '<title id="title">GPT-2 attention-head profiles</title>',
           '<desc id="desc">Each of the 144 dots is the profile of an attention head, projected from the three-simplex.</desc>',
           '<path d="M68 176 L240 40 L373 202 Z" fill="#edf0e9"/>']
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = VERTICES[i], VERTICES[j]
            dashed = ' stroke-dasharray="4 5"' if (i, j) == (1, 2) else ''
            svg.append(f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" fill="none" stroke="#87998e" stroke-width="1.1"{dashed}/>')
    for x, y in points:
        svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.4" fill="#276c62" fill-opacity=".68"/>')
    for label, x, y in [('a',245,318),('b',47,181),('c',394,207),('d',240,24)]:
        svg.append(f'<text x="{x}" y="{y}" fill="#566a5f" font-family="Georgia,serif" font-style="italic" font-size="17" text-anchor="middle">{label}</text>')
    svg.append('</svg>')
    (output / 'head-profiles.svg').write_text('\n'.join(svg) + '\n')

    from PIL import Image, ImageDraw, ImageFont
    def font(size, serif=False):
        candidates = ([Path('/System/Library/Fonts/Supplemental/Georgia.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')]
                      if serif else [Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')])
        return ImageFont.truetype(str(next(p for p in candidates if p.exists())), size)
    im = Image.new('RGB', (1200, 630), '#f8f7f3');d = ImageDraw.Draw(im)
    d.text((70,65), 'MATHEMATICS OF ATTENTION', font=font(17), fill='#65706b')
    d.text((66,150), 'On attention heads', font=font(54,True), fill='#252f2d')
    d.text((66,220), 'and bilinear forms', font=font(54,True), fill='#252f2d')
    d.text((70,322), 'Andrew O’Desky', font=font(23), fill='#65706b')
    d.text((70,460), 'Interactive plots · Lean blueprint · Code & data', font=font(20), fill='#276c62')
    def xy(p): return (int(680 + p[0]), int(95 + p[1]))
    d.polygon([xy(VERTICES[i]) for i in [1,3,2]], fill='#edf0e9')
    for i in range(4):
        for j in range(i+1,4):d.line([xy(VERTICES[i]),xy(VERTICES[j])],fill='#87998e',width=2)
    for p in points:
        x,y=xy(p);d.ellipse((x-3,y-3,x+3,y+3),fill='#276c62')
    d.line((70,552,1130,552),fill='#dedfd7',width=1)
    d.text((70,575), 'aodesky.github.io/attention-bilinear-forms',font=font(17),fill='#65706b')
    im.save(output/'social-preview.png', optimize=True)
    print('Generated homepage artwork from 144 GPT-2 profiles.')


if __name__ == '__main__':
    main()
