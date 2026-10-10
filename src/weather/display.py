"""Tables, prose and charts, kept away from everything that fetches or parses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from src.weather.codes import precipitating

__all__ = ["Comparison", "FIGURES", "INK", "chart_week", "chart_comparison",
           "compare", "house_style", "readable_table", "save", "summarise"]

FIGURES = Path(__file__).resolve().parents[2] / "figures"

# Series colours, legible in greyscale.
INK = ("#17415c", "#9c4312", "#2f6b2f", "#5d3b8c")


def house_style() -> None:
    """Shared chart settings; applied once per notebook."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": (0, (1, 3)),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9, "legend.frameon": False,
        "lines.markersize": 4,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(INK))
    FIGURES.mkdir(exist_ok=True)


def save(figure: plt.Figure, filename: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    target = FIGURES / filename
    figure.savefig(target)
    return target


def readable_table(frame: pd.DataFrame) -> pd.DataFrame:
    """A display copy: spelled-out dates, units in the headings."""
    shown = frame.copy()
    shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%a %d %b")
    return shown.rename(columns={
        "date": "Day", "max_c": "Max (°C)", "min_c": "Min (°C)",
        "rain_mm": "Rain (mm)", "wind_kmh": "Wind (km/h)", "outlook": "Outlook",
    }).drop(columns=["code"], errors="ignore")


def summarise(frame: pd.DataFrame, place: str, licence: str = "") -> str:
    """A short prose reading of the forecast, for the text part of task B4."""
    if frame.empty:
        return f"No forecast is available for {place}."
    wet = int(sum(precipitating(code) for code in frame["code"]))
    warmest = frame.loc[frame["max_c"].idxmax()]
    coldest = frame.loc[frame["min_c"].idxmin()]
    blowiest = frame.loc[frame["wind_kmh"].idxmax()]

    lines = [
        f"{len(frame)} days for {place}, {frame.iloc[0]['date']} to "
        f"{frame.iloc[-1]['date']}.",
        f"Maxima run {frame['max_c'].min():.1f}-{frame['max_c'].max():.1f} °C, "
        f"minima {frame['min_c'].min():.1f}-{frame['min_c'].max():.1f} °C.",
        f"Warmest: {warmest['date']} at {warmest['max_c']:.1f} °C. "
        f"Coolest night: {coldest['date']} at {coldest['min_c']:.1f} °C.",
        f"{frame['rain_mm'].sum():.1f} mm of rain in total"
        + (f", on {wet} of {len(frame)} days." if wet
           else ", with no day showing precipitation."),
        f"Strongest wind {blowiest['wind_kmh']:.1f} km/h on {blowiest['date']}.",
    ]
    if "outlook" in frame and frame["outlook"].notna().any():
        usual = frame["outlook"].mode()
        if len(usual):
            lines.append(f"Most frequent outlook: {usual.iloc[0].lower()}.")
    if licence:
        lines.append(f"Source: {licence}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Comparison:
    """Two providers' tables, joined on date, with the gaps measured."""

    table: pd.DataFrame
    first: str
    second: str

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (f"Comparison({self.first} vs {self.second}, "
                f"{len(self)} shared days)")

    def measures(self) -> list[str]:
        return [column.removesuffix(" gap") for column in self.table.columns
                if column.endswith(" gap")]

    def gap(self, measure: str) -> pd.Series:
        column = f"{measure} gap"
        if column not in self.table.columns:
            raise KeyError(f"no measure {measure!r}; have {self.measures()}")
        return self.table[column]

    def overview(self) -> pd.DataFrame:
        """Mean and worst absolute gap for each measure."""
        return pd.DataFrame([{
            "measure": measure,
            "mean |gap|": self.gap(measure).abs().mean(),
            "worst |gap|": self.gap(measure).abs().max(),
        } for measure in self.measures()]).set_index("measure")

    def reads_as(self, measure: str = "max_c", tolerance: float = 2.0) -> str:
        worst = self.gap(measure).abs().max()
        verb = "agree" if worst <= tolerance else "disagree"
        return (f"on {measure} the two sources {verb}: worst gap {worst:.2f} "
                f"against a {tolerance} tolerance")


def compare(first: pd.DataFrame, second: pd.DataFrame,
            first_name: str = "Open-Meteo",
            second_name: str = "MET Norway") -> Comparison:
    """Join two forecasts on date and measure where they differ.

    Inner join: an outer one would report different horizons as differences.
    """
    left, right = first.set_index("date"), second.set_index("date")
    shared = left.index.intersection(right.index)
    if not len(shared):
        raise ValueError("the two forecasts share no dates")
    out = pd.DataFrame(index=shared)
    for measure in ("max_c", "min_c", "rain_mm", "wind_kmh"):
        out[f"{measure} ({first_name})"] = left.loc[shared, measure]
        out[f"{measure} ({second_name})"] = right.loc[shared, measure]
        out[f"{measure} gap"] = (left.loc[shared, measure]
                                 - right.loc[shared, measure]).round(2)
    out.index.name = "date"
    return Comparison(out, first_name, second_name)


def chart_week(frame: pd.DataFrame, place: str,
               figsize: tuple[float, float] = (11, 5)) -> plt.Figure:
    """Temperature range as a band, rainfall on a second axis."""
    if frame.empty:
        raise ValueError("there is nothing to chart")
    figure, ax = plt.subplots(figsize=figsize)
    days = pd.to_datetime(frame["date"])

    ax.fill_between(days, frame["min_c"], frame["max_c"], color=INK[0],
                    alpha=0.16, label="daily range")
    ax.plot(days, frame["max_c"], "o-", color=INK[0], label="maximum (°C)")
    ax.plot(days, frame["min_c"], "o--", color=INK[1], label="minimum (°C)")
    ax.set_xlabel("date")
    ax.set_ylabel("temperature (°C)")
    ax.set_title(f"{len(frame)}-day forecast for {place}")

    rain = ax.twinx()
    rain.bar(days, frame["rain_mm"], width=0.55, color=INK[2], alpha=0.35,
             label="rainfall (mm)")
    rain.set_ylabel("rainfall (mm)")
    rain.set_ylim(0, max(float(frame["rain_mm"].max()) * 2.2, 1.0))
    rain.grid(False)

    lines, labels = ax.get_legend_handles_labels()
    bars, bar_labels = rain.get_legend_handles_labels()
    ax.legend(lines + bars, labels + bar_labels, loc="upper left", fontsize=8,
              ncol=2)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    return figure


def chart_comparison(comparison: Comparison,
                     figsize: tuple[float, float] = (12, 7)) -> plt.Figure:
    """Both providers on temperature and rainfall, gaps shaded."""
    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    days = pd.to_datetime(comparison.table.index)
    for ax, measure, label in ((axes[0], "max_c", "daily maximum (°C)"),
                               (axes[1], "rain_mm", "rainfall (mm)")):
        a = comparison.table[f"{measure} ({comparison.first})"]
        b = comparison.table[f"{measure} ({comparison.second})"]
        ax.plot(days, a, "o-", color=INK[0], label=comparison.first)
        ax.plot(days, b, "s--", color=INK[1], label=comparison.second)
        ax.fill_between(days, a, b, color="0.5", alpha=0.2, label="gap")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    axes[0].set_title("Two independent models on the same point")
    axes[1].set_xlabel("date")
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    return figure
