#!/usr/bin/env python3
"""Interactive viewer for the poker raise-size action abstraction.

This mirrors ``RaiseMixtureHead`` and ``PokerPolicyAdapter`` for a single
decision row.  The two continuous raise modes are discretized logistic
distributions; the all-in probability is an atom on the largest legal raise.

Usage::

    python tools/plot_raise_distribution.py

Requires Matplotlib (the training package itself does not depend on it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_TICKS = 32


HTML_TEMPLATE = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Raise distribution explorer</title>
<style>
body { font: 14px sans-serif; max-width: 1100px; margin: 24px auto; }
canvas { width: 100%; height: 430px; border: 1px solid #ddd; }
.row { display: grid; grid-template-columns: 170px 1fr 60px; gap: 10px; margin: 9px 0; }
button { margin: 12px 8px 12px 0; padding: 6px 14px; }
</style></head><body>
<h2>Legal raise distribution: two logistic modes + all-in atom</h2>
<canvas id="plot" width="1050" height="430"></canvas>
<div id="controls"></div><button id="reset">Reset</button>
<button id="sweep">Pause sweep</button>
<script>
const N=32, defaults=[.50,.28,.10,.72,.10,.15,.10,.90];
const specs=[['mode 1 weight',0,1],['mode 1 location',0,1],['mode 1 scale',.02,.50],
 ['mode 2 location',0,1],['mode 2 scale',.02,.50],['all-in iota',0,1],
 ['legal minimum',0,.90],['legal maximum',.10,1]];
const controls=document.getElementById('controls'), sliders=[];
specs.forEach((s,i)=>{let d=document.createElement('div'); d.className='row';
 d.innerHTML=`<label>${s[0]}</label><input type="range" min="${s[1]}" max="${s[2]}" step=".01" value="${defaults[i]}"><output></output>`;
 controls.appendChild(d); let input=d.querySelector('input'), out=d.querySelector('output');
 input.oninput=()=>draw(); sliders.push(input);});
function logistic(x,mu,scale){return 1/(1+Math.exp(-(x-mu)/scale));}
function draw(){
 let v=sliders.map(x=>+x.value), [w,mu1,s1,mu2,s2,iota,a,b]=v, lo=Math.min(a,b), hi=Math.max(a,b);
 sliders.forEach((x,i)=>x.parentElement.querySelector('output').value=(+x.value).toFixed(2));
 let p=[], legal=[], lsum=0, qsum=0;
 for(let i=0;i<N;i++){let x=(i+.5)/N, l=logistic((i+1)/N,mu1,s1)-logistic(i/N,mu1,s1),
  q=logistic((i+1)/N,mu2,s2)-logistic(i/N,mu2,s2); lsum+=l; qsum+=q; legal[i]=x>=lo&&x<=hi; p[i]=[l,q];}
 for(let i=0;i<N;i++) p[i]=w*p[i][0]/lsum+(1-w)*p[i][1]/qsum;
 let z=p.reduce((s,x,i)=>s+(legal[i]?x:0),0); p=p.map((x,i)=>legal[i]?(1-iota)*x/z:0);
 let last=legal.reduce((last,x,i)=>x?i:last,-1); if(last<0) last=Math.max(0,Math.min(N-1,Math.round((lo+hi)*N/2)));
 p[last]=(p[last]||0)+iota; let c=document.getElementById('plot'),ctx=c.getContext('2d'), W=c.width,H=c.height;
 ctx.clearRect(0,0,W,H); ctx.font='12px sans-serif'; ctx.strokeStyle='#999'; ctx.beginPath(); ctx.moveTo(45,H-45);ctx.lineTo(W-15,H-45);ctx.stroke();
 let ymax=Math.max(.05,...p)*1.15; for(let i=0;i<N;i++){let x=48+i*(W-70)/N, h=p[i]/ymax*(H-75);
  ctx.fillStyle=legal[i]?'#f58518':'#d9d9d9';ctx.fillRect(x,H-45-h,(W-75)/N-2,h);ctx.fillStyle='#333';
  if(i%4===0)ctx.fillText(((i+.5)/N).toFixed(2),x,H-25);}
 ctx.fillText('probability',8,18);ctx.fillText('normalized raise-to amount (largest legal tick = all-in)',W/2-150,H-5);
}
document.getElementById('reset').onclick=()=>{sliders.forEach((x,i)=>x.value=defaults[i]);draw();};
let running=true, frame=0, sweepButton=document.getElementById('sweep');
sweepButton.onclick=()=>{running=!running;sweepButton.textContent=running?'Pause sweep':'Resume sweep';};
setInterval(()=>{if(running){let t=(frame++%240)/239;sliders[0].value=.02+.96*(t<.5?2*t:2-2*t);draw();}},45); draw();
</script></body></html>'''


def write_html(path: Path) -> None:
    path.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Wrote interactive plot to {path.resolve()}")


def discretized_logistic(mu: float, scale: float, n_ticks: int) -> np.ndarray:
    """Return normalized logistic mass for equally spaced [0, 1] bins."""

    edges = np.linspace(0.0, 1.0, n_ticks + 1)
    cdf = lambda x: 1.0 / (1.0 + np.exp(-(x - mu) / scale))
    mass = cdf(edges[1:]) - cdf(edges[:-1])
    return mass / mass.sum()


def legal_raise_distribution(
    weight_1: float,
    mu_1: float,
    scale_1: float,
    mu_2: float,
    scale_2: float,
    allin_probability: float,
    legal_min: float,
    legal_max: float,
    n_ticks: int = N_TICKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose the legal raise-tick probabilities.

    ``legal_min`` and ``legal_max`` are normalized raise-to amounts.  The
    final legal bin is the all-in event, matching the repository adapter.
    """

    w = np.array([weight_1, 1.0 - weight_1])
    continuous = w[0] * discretized_logistic(mu_1, scale_1, n_ticks)
    continuous += w[1] * discretized_logistic(mu_2, scale_2, n_ticks)
    centers = (np.arange(n_ticks) + 0.5) / n_ticks
    legal = (centers >= legal_min) & (centers <= legal_max)
    if not legal.any():
        # The sliders can be positioned between adjacent bins. Keep one legal
        # event visible rather than creating an invalid empty action set.
        nearest = int(np.argmin(np.abs(centers - (legal_min + legal_max) / 2.0)))
        legal[nearest] = True
    continuous[~legal] = 0.0
    continuous /= continuous.sum()

    result = (1.0 - allin_probability) * continuous
    allin_index = np.flatnonzero(legal)[-1]
    result[allin_index] += allin_probability
    return centers, result, legal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html", type=Path, metavar="PATH",
        help="write a self-contained browser plot instead of opening Matplotlib",
    )
    args = parser.parse_args()
    if args.html is not None:
        write_html(args.html)
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, Slider
    except ImportError as exc:
        raise SystemExit(
            "This interactive demo requires Matplotlib; install it with "
            "`/home/pablob/.venvs/poker/bin/python -m pip install matplotlib`, "
            "or use `--html raise_distribution.html` on a headless server."
        ) from exc

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.40, top=0.90)
    centers = (np.arange(N_TICKS) + 0.5) / N_TICKS
    bars = ax.bar(centers, np.zeros(N_TICKS), width=0.027, color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized raise-to amount (largest legal tick = all-in)")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Legal raise distribution: two discretized logistic modes + all-in atom")

    # Sliders are deliberately named after the model head's parameters.
    slider_specs = [
        ("mode 1 weight", 0.0, 1.0, 0.50),
        ("mode 1 location", 0.0, 1.0, 0.28),
        ("mode 1 scale", 0.02, 0.50, 0.10),
        ("mode 2 location", 0.0, 1.0, 0.72),
        ("mode 2 scale", 0.02, 0.50, 0.10),
        ("all-in iota", 0.0, 1.0, 0.15),
        ("legal minimum", 0.0, 0.90, 0.10),
        ("legal maximum", 0.10, 1.0, 0.90),
    ]
    sliders: list[Slider] = []
    for row, (label, lower, upper, initial) in enumerate(slider_specs):
        y = 0.34 - row * 0.035
        slider_axis = fig.add_axes((0.24, y, 0.64, 0.022))
        slider = Slider(slider_axis, label, lower, upper, valinit=initial)
        sliders.append(slider)

    weight, loc1, scale1, loc2, scale2, iota, legal_min, legal_max = sliders

    def redraw(_value: float | None = None) -> None:
        # Keep the legal interval nonempty if the sliders cross.
        lo, hi = sorted((legal_min.val, legal_max.val))
        _, probabilities, legal = legal_raise_distribution(
            weight.val, loc1.val, scale1.val, loc2.val, scale2.val,
            iota.val, lo, hi,
        )
        for bar, probability, is_legal in zip(bars, probabilities, legal):
            bar.set_height(probability)
            bar.set_color("#f58518" if is_legal else "#d9d9d9")
        ax.set_ylim(0, max(0.05, float(probabilities.max()) * 1.18))
        ax.set_title(
            f"Legal raise distribution  |  continuous mass = {1 - iota.val:.2f}"
            f"  |  all-in atom = {iota.val:.2f}"
        )
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(redraw)

    reset_axis = fig.add_axes((0.10, 0.035, 0.12, 0.045))
    reset = Button(reset_axis, "Reset")

    def reset_sliders(_event: object) -> None:
        for slider in sliders:
            slider.reset()

    reset.on_clicked(reset_sliders)

    play_axis = fig.add_axes((0.24, 0.035, 0.12, 0.045))
    play = Button(play_axis, "Sweep mode 1")
    sweep_running = True

    def toggle_sweep(_event: object) -> None:
        nonlocal sweep_running
        if animation.event_source is None:
            return
        if sweep_running:
            animation.event_source.stop()
            play.label.set_text("Resume sweep")
        else:
            animation.event_source.start()
            play.label.set_text("Pause sweep")
        sweep_running = not sweep_running
        fig.canvas.draw_idle()

    play.on_clicked(toggle_sweep)

    def animate(frame: int) -> None:
        # A small periodic sweep makes the distribution visibly animated while
        # every slider remains live and can take over at any time.
        phase = (frame % 120) / 119.0
        value = 0.02 + 0.96 * (phase if (frame // 120) % 2 == 0 else 1.0 - phase)
        weight.set_val(value)

    animation = FuncAnimation(fig, animate, interval=45, blit=False, cache_frame_data=False)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
