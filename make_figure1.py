"""
Stage 6: Figure 1 (fig:censoring) from the cleaned record, drawn as vector PDF with
reportlab (pure Python; no compiled plotting backend is required).

(a) daily published demand against daily served generation, 2016-2021 vs 2022-2024
(b) hourly probability of reported shedding, 2022-2024, 18:00-21:00 peak window shaded
(c) load-duration curves of hourly generation, 2019, 2021, 2023

Inputs   results/panel_daily.csv, results/hourly_clean.csv
Output   figs/fig1_censoring.pdf
"""
import csv, os
import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle, PolyLine, Group
from reportlab.graphics import renderPDF
from reportlab.lib.colors import Color, black, white

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
GREY = Color(0.29, 0.29, 0.29); RED = Color(0.75, 0.22, 0.17); LIGHT = Color(0.50, 0.55, 0.55)
FONT = "Helvetica"


def load(path, datecol, cols, dropna=False):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    dates = np.array([r[datecol][:19].replace(" ", "T") for r in rows], dtype="datetime64[s]")
    arr = {c: np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows]) for c in cols}
    if dropna:
        ok = np.all([~np.isnan(arr[c]) for c in cols], axis=0)
        dates = dates[ok]; arr = {c: v[ok] for c, v in arr.items()}
    year = dates.astype("datetime64[Y]").astype(int) + 1970
    hour = ((dates - dates.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int)
    return {"year": year, "hour": hour, **arr}


class Panel:
    """A tiny axes helper: data -> page coordinates, with ticks and labels."""

    def __init__(self, d, x0, y0, w, h, xlim, ylim, title, xlabel, ylabel):
        self.d, self.x0, self.y0, self.w, self.h, self.xlim, self.ylim = d, x0, y0, w, h, xlim, ylim
        d.add(Line(x0, y0, x0 + w, y0, strokeColor=black, strokeWidth=0.6))
        d.add(Line(x0, y0, x0, y0 + h, strokeColor=black, strokeWidth=0.6))
        d.add(String(x0, y0 + h + 6, title, fontName=FONT + "-Bold", fontSize=8))
        d.add(String(x0 + w / 2, y0 - 22, xlabel, fontName=FONT, fontSize=7, textAnchor="middle"))
        g = Group(String(0, 0, ylabel, fontName=FONT, fontSize=7, textAnchor="middle"))
        g.translate(x0 - 26, y0 + h / 2); g.rotate(90); d.add(g)

    def X(self, x): return self.x0 + (np.asarray(x) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w
    def Y(self, y): return self.y0 + (np.asarray(y) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def xticks(self, vals, fmt="{:g}"):
        for v in vals:
            x = float(self.X(v)); self.d.add(Line(x, self.y0, x, self.y0 - 2.5, strokeColor=black, strokeWidth=0.6))
            self.d.add(String(x, self.y0 - 10, fmt.format(v), fontName=FONT, fontSize=6.5, textAnchor="middle"))

    def yticks(self, vals, fmt="{:g}"):
        for v in vals:
            y = float(self.Y(v)); self.d.add(Line(self.x0, y, self.x0 - 2.5, y, strokeColor=black, strokeWidth=0.6))
            self.d.add(String(self.x0 - 4, y - 2.2, fmt.format(v), fontName=FONT, fontSize=6.5, textAnchor="end"))


def main():
    P = load(os.path.join(RES, "panel_daily.csv"), "date", ["G", "Dpub"])
    H = load(os.path.join(RES, "hourly_clean.csv"), "dt", ["generation_mw", "load_shedding"], dropna=True)

    W, Hh = 520, 190
    d = Drawing(W, Hh)
    pw, ph, y0 = 135, 120, 40
    xs = [38, 38 + pw + 45, 38 + 2 * (pw + 45)]

    # (a) published demand vs served generation, GWh/day
    G, D = P["G"] / 1e3, P["Dpub"] / 1e3
    lo, hi = float(np.nanmin(G)) * 0.95, float(np.nanmax(D)) * 1.02
    a = Panel(d, xs[0], y0, pw, ph, (lo, hi), (lo, hi), "(a) Published demand vs supply", "Served generation (GWh/day)", "Published demand (GWh/day)")
    for yrs, col in [((2016, 2021), GREY), ((2022, 2024), RED)]:
        m = (P["year"] >= yrs[0]) & (P["year"] <= yrs[1])
        for x, y in zip(a.X(G[m]), a.Y(D[m])):
            d.add(Circle(float(x), float(y), 0.7, fillColor=col, strokeColor=None, fillOpacity=0.35))
    d.add(Line(float(a.X(lo)), float(a.Y(lo)), float(a.X(hi)), float(a.Y(hi)), strokeColor=black, strokeWidth=0.6, strokeDashArray=[2, 2]))
    tk = np.arange(np.ceil(lo / 50) * 50, hi, 50); a.xticks(tk); a.yticks(tk)
    d.add(Circle(xs[0] + 8, y0 + ph - 8, 2, fillColor=GREY, strokeColor=None)); d.add(String(xs[0] + 13, y0 + ph - 10, "2016-2021", fontName=FONT, fontSize=6.5))
    d.add(Circle(xs[0] + 8, y0 + ph - 18, 2, fillColor=RED, strokeColor=None)); d.add(String(xs[0] + 13, y0 + ph - 20, "2022-2024", fontName=FONT, fontSize=6.5))

    # (b) Pr(reported shedding) by hour label, 2022-2024
    m = (H["year"] >= 2022) & (H["year"] <= 2024)
    hrs = np.arange(24)
    p = np.array([(H["load_shedding"][m & (H["hour"] == h)] > 0).mean() if (m & (H["hour"] == h)).any() else np.nan for h in hrs])
    ymax = float(np.nanmax(p)) * 1.15
    b = Panel(d, xs[1], y0, pw, ph, (-0.5, 23.5), (0, ymax), "(b) Shedding by hour, 2022-2024", "Hour of day (label)", "Pr(reported shedding)")
    d.add(Rect(float(b.X(17.5)), y0, float(b.X(21.5) - b.X(17.5)), ph, fillColor=RED, strokeColor=None, fillOpacity=0.12))
    for h_, v in zip(hrs, p):
        if not np.isnan(v):
            d.add(Rect(float(b.X(h_ - 0.4)), y0, float(b.X(0.4) - b.X(-0.4)), float(b.Y(v) - y0), fillColor=GREY, strokeColor=None))
    d.add(String(float(b.X(19.5)), float(b.Y(np.nanmax(p))) + 3, "peak", fontName=FONT, fontSize=6.5, textAnchor="middle", fillColor=RED))
    b.xticks(range(0, 24, 4)); b.yticks(np.arange(0, ymax, 0.2), "{:.1f}")

    # (c) load-duration curves, GW
    gmax = float(H["generation_mw"].max()) / 1e3 * 1.05
    c = Panel(d, xs[2], y0, pw, ph, (0, 100), (0, gmax), "(c) Load-duration curves", "Share of hours (%)", "Generation (GW)")
    for y_, col, k in [(2019, LIGHT, 0), (2021, GREY, 1), (2023, RED, 2)]:
        g = np.sort(H["generation_mw"][H["year"] == y_])[::-1] / 1e3
        idx = np.linspace(0, len(g) - 1, 300).astype(int)
        pts = []
        for xq, gq in zip(np.linspace(0, 100, 300), g[idx]):
            pts += [float(c.X(xq)), float(c.Y(gq))]
        d.add(PolyLine(pts, strokeColor=col, strokeWidth=1.1))
        d.add(Line(xs[2] + pw - 40, y0 + 14 + 9 * k, xs[2] + pw - 30, y0 + 14 + 9 * k, strokeColor=col, strokeWidth=1.1))
        d.add(String(xs[2] + pw - 27, y0 + 12 + 9 * k, str(y_), fontName=FONT, fontSize=6.5))
    c.xticks(range(0, 101, 25)); c.yticks(range(0, int(gmax) + 1, 4))

    renderPDF.drawToFile(d, os.path.join(FIG, "fig1_censoring.pdf"))
    print("wrote figs/fig1_censoring.pdf")


if __name__ == "__main__":
    main()
