import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import questionary
from PIL import Image, ImageCms
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

HEX_RE = re.compile(r"^#?([\da-fA-F]{3}|[\da-fA-F]{6})$")
RGB_RE = re.compile(r"rgb\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)$")
CMYK_RE = re.compile(r"cmyk\(\s*([\d%]+),\s*([\d%]+),\s*([\d%]+),\s*([\d%]+)\s*\)$")

DEF_SRGB = "public/sRGB2014.icc"
DEF_CMYK = "public/ISOcoated_v2_eci.icc"
DEF_PARQ = "public/colors.parquet"
DEF_BATCH_FILE = Path("input/colors.txt")

INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC
FLAGS = ImageCms.Flags.BLACKPOINTCOMPENSATION


def clamp(v):
    return max(0, min(255, int(round(v))))


def _fmt_ok(x, dec=2):
    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    return round(x, dec)


def percent_to_255(v):
    v = v.strip()
    if v.endswith("%"):
        return clamp(float(v[:-1]) / 100 * 255)
    f = float(v)
    if f <= 1:
        return clamp(f * 255)
    if f <= 100:
        return clamp(f / 100 * 255)
    return clamp(f)


def parse_color(t):
    t = t.strip()
    m = HEX_RE.match(t)
    if m:
        h = m.group(1)
        if len(h) == 3:
            r, g, b = [int(c * 2, 16) for c in h]
        else:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return "rgb", (r, g, b)

    m = RGB_RE.match(t)
    if m:
        return "rgb", tuple(clamp(int(x)) for x in m.groups())

    m = CMYK_RE.match(t)
    if m:
        return "cmyk", tuple(percent_to_255(x) for x in m.groups())

    raise ValueError(f"Invalid color input: {t}")


def load_icc(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    return ImageCms.ImageCmsProfile(str(p))


def transform_color(src_rgb, src_profile, dst_profile, src_mode, dst_mode):
    t = ImageCms.buildTransformFromOpenProfiles(
        src_profile,
        dst_profile,
        src_mode,
        dst_mode,
        renderingIntent=INTENT,
        flags=FLAGS,
    )
    img = Image.new(src_mode, (1, 1), src_rgb)
    return ImageCms.applyTransform(img, t).getpixel((0, 0))  # pyright: ignore[reportOptionalMemberAccess]


def srgb_to_cmyk(rgb, sr, cm):
    return transform_color(rgb, sr, cm, "RGB", "CMYK")


def cmyk_to_srgb(cmyk, cm, sr):
    return transform_color(cmyk, cm, sr, "CMYK", "RGB")


def srgb_lin(v):
    v /= 255
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def rgb_to_oklch(rgb):
    r, g, b = map(srgb_lin, rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b  # noqa: E741
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_, m_, s_ = np.cbrt([l, m, s])
    L = 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_

    C = (a * a + b * b) ** 0.5
    h = float(np.degrees(np.arctan2(b, a)) % 360)

    return {"L": float(L), "C": float(C), "h": h}


def rel_lum(rgb):
    r, g, b = map(srgb_lin, rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def find_color_name(hexv, oklch, parquet_df):
    if parquet_df is None or parquet_df.empty:
        return None, False

    row = parquet_df[parquet_df["hex_norm"] == hexv.upper()]
    if not row.empty:
        return row.iloc[0]["name"], False

    Ls = parquet_df["L"].to_numpy(dtype=float)
    Cs = parquet_df["C"].to_numpy(dtype=float)
    hs = parquet_df["h"].to_numpy(dtype=float)

    dL = Ls - oklch["L"]
    dC = Cs - oklch["C"]
    dh_raw = np.abs(hs - oklch["h"])
    dh = np.minimum(dh_raw, 360.0 - dh_raw)
    dist = np.sqrt(dL * dL + dC * dC + dh * dh)

    idx = int(np.argmin(dist))
    return parquet_df.iloc[idx]["name"], True


def contrast(a, b):
    L1, L2 = rel_lum(a), rel_lum(b)
    if L1 < L2:
        L1, L2 = L2, L1
    return (L1 + 0.05) / (L2 + 0.05)


def save_batch_json(results):
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = outdir / f"color-info-{ts}.json"

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✔ Saved batch results to[/green] {outfile}")


def load_parquet(path):
    p = Path(path)
    if not p.exists():
        return None

    df = pd.read_parquet(str(p))

    if "hex" not in df.columns or "bestContrast" not in df.columns:
        return None

    df = df.copy()
    df["hex_norm"] = (
        df["hex"]
        .astype(str)
        .str.upper()
        .str.replace("#", "", regex=False)
        .apply(lambda x: "#" + x)
    )

    if not all(col in df.columns for col in ("L", "C", "h")):
        hexs = df["hex_norm"].to_numpy(dtype=str)
        n = len(hexs)
        rgb_arr = np.empty((n, 3), dtype=np.float64)
        for i, h in enumerate(hexs):
            rgb_arr[i, 0] = int(h[1:3], 16)
            rgb_arr[i, 1] = int(h[3:5], 16)
            rgb_arr[i, 2] = int(h[5:7], 16)
        # vectorized OKLCH
        _SRGB_TO_LMS = np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        )
        a = rgb_arr / 255.0
        mask = a <= 0.04045
        lin = np.empty_like(a)
        lin[mask] = a[mask] / 12.92
        lin[~mask] = ((a[~mask] + 0.055) / 1.055) ** 2.4
        lms = lin @ _SRGB_TO_LMS.T
        lms = np.clip(lms, 0, None)
        lms_cbrt = np.cbrt(lms)
        Ls = (
            0.2104542553 * lms_cbrt[:, 0]
            + 0.7936177850 * lms_cbrt[:, 1]
            - 0.0040720468 * lms_cbrt[:, 2]
        )
        a_ = (
            1.9779984951 * lms_cbrt[:, 0]
            - 2.4285922050 * lms_cbrt[:, 1]
            + 0.4505937099 * lms_cbrt[:, 2]
        )
        b_ = (
            0.0259040371 * lms_cbrt[:, 0]
            + 0.7827717662 * lms_cbrt[:, 1]
            - 0.8086757660 * lms_cbrt[:, 2]
        )
        Cs = np.hypot(a_, b_)
        hs = np.degrees(np.arctan2(b_, a_)) % 360
        df["L"] = Ls
        df["C"] = Cs
        df["h"] = hs

    return df


def print_color_bar(hexv):
    console.print(
        Panel("", width=40, height=1, style=f"on {hexv}", border_style=hexv, padding=0)
    )


def display_result(hexv, rgb, cmyk_pct, oklch, final_choice, name, was_closest):
    c, m, y, k = [round(x) for x in cmyk_pct]
    L = _fmt_ok(oklch["L"])
    C = _fmt_ok(oklch["C"])
    h = _fmt_ok(oklch["h"])

    name_val = (
        "Unknown" if name is None else name + (" [Closest]" if was_closest else "")
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold white", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Name", name_val)
    table.add_row("HEX", hexv)
    table.add_row("RGB", f"{rgb[0]}, {rgb[1]}, {rgb[2]}")
    table.add_row("CMYK", f"{c}, {m}, {y}, {k}")
    table.add_row("OKLCH", f"{L}, {C}, {h}")
    table.add_row("Contrast", final_choice)

    console.print(table)


def process_single(color_text, cm, sr, pq):
    fmt, val = parse_color(color_text)

    rgb = cmyk_to_srgb(val, cm, sr) if fmt == "cmyk" else val
    cmyk255 = val if fmt == "cmyk" else srgb_to_cmyk(rgb, sr, cm)

    hexv = "#{:02X}{:02X}{:02X}".format(*rgb)  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
    cmyk_pct = [(x / 255) * 100 for x in cmyk255]  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
    oklch = rgb_to_oklch(rgb)
    name, was_closest = find_color_name(hexv, oklch, pq)

    white, black = (255, 255, 255), (0, 0, 0)
    suggested = "white" if contrast(rgb, white) > contrast(rgb, black) else "black"

    override = None
    if pq is not None:
        row = pq[pq["hex_norm"] == hexv.upper()]
        if not row.empty:
            override = row.iloc[0]["bestContrast"].lower()

    final = override if override in ("white", "black") else suggested

    console.print("\n[bold cyan]Color Preview[/bold cyan]")
    print_color_bar(hexv)

    console.print("\n[bold cyan]Color Analysis[/bold cyan]")
    display_result(hexv, rgb, cmyk_pct, oklch, final, name, was_closest)


def run_batch(batch_path: Path, cm, sr, pq):
    if not batch_path.exists():
        console.print(f"[red]Batch file not found:[/red] {batch_path}")
        return

    lines = [l.strip() for l in batch_path.read_text().splitlines() if l.strip()]  # noqa: E741
    results = []

    for line in lines:
        try:
            fmt, val = parse_color(line)
        except Exception:
            results.append({"input": line, "error": "Invalid color"})
            continue

        rgb = cmyk_to_srgb(val, cm, sr) if fmt == "cmyk" else val
        cmyk255 = val if fmt == "cmyk" else srgb_to_cmyk(rgb, sr, cm)

        hexv = "#{:02X}{:02X}{:02X}".format(*rgb)  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
        cmyk_pct = [(x / 255) * 100 for x in cmyk255]  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
        oklch = rgb_to_oklch(rgb)
        name, was_closest = find_color_name(hexv, oklch, pq)

        white, black = (255, 255, 255), (0, 0, 0)
        suggested = "white" if contrast(rgb, white) > contrast(rgb, black) else "black"

        override = None
        if pq is not None:
            row = pq[pq["hex_norm"] == hexv.upper()]
            if not row.empty:
                override = row.iloc[0]["bestContrast"].lower()

        final = override if override in ("white", "black") else suggested

        results.append(
            {
                "hex": hexv,
                "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
                "cmyk": [round(x) for x in cmyk_pct],
                "oklch": {
                    "L": _fmt_ok(oklch["L"]),
                    "C": _fmt_ok(oklch["C"]),
                    "h": _fmt_ok(oklch["h"]),
                },
                "name": name,
                "closest_match": bool(was_closest),
                "contrast": final,
            }
        )

    save_batch_json(results)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--default", action="store_true")
    parser.add_argument("--batch", nargs="?", const=True)
    args = parser.parse_args()

    if args.default:
        cm_path = DEF_CMYK
        sr_path = DEF_SRGB
        pq_path = DEF_PARQ
    else:
        cm_path = questionary.text("CMYK profile:", default=DEF_CMYK).ask()
        sr_path = questionary.text("sRGB profile:", default=DEF_SRGB).ask()
        pq_path = questionary.text("Parquet lookup:", default=DEF_PARQ).ask()

    cm = load_icc(cm_path)
    sr = load_icc(sr_path)
    pq = load_parquet(pq_path)

    if args.batch is not None:
        if args.default:
            batch_path = DEF_BATCH_FILE
        else:
            batch_path_input = questionary.text(
                "Batch file:", default=str(DEF_BATCH_FILE)
            ).ask()
            batch_path = Path(batch_path_input)

        run_batch(batch_path, cm, sr, pq)
        return

    color_input = questionary.text("Enter color:").ask()
    process_single(color_input, cm, sr, pq)


if __name__ == "__main__":
    main()
