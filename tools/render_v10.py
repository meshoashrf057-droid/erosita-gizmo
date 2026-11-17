#!/usr/bin/env python3
"""Portable entry point for the project's unchanged v10_log yt plotting functions.

This wrapper selects single-file snapshots, isolates caches, and renders the
requested fields of view. It does not change the scientific plot algorithms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "erosita_xray" / "erosita_bubble_xray_yt_v10_log.py"
PATTERN = re.compile(r"snapshot_(\d+)\.hdf5$")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fovs", type=int, nargs="+", choices=(44, 200),
                        default=[44, 200], help="Full image widths in kpc.")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--indices", type=int, nargs="+",
                        help="Explicit snapshot numbers, for example 8 9 10.")
    select.add_argument("--start", type=int, help="First snapshot number, inclusive.")
    parser.add_argument("--stop", type=int, help="Last snapshot number, inclusive.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every Nth file after start/stop selection.")
    parser.add_argument("--reference-ic", type=Path,
                        help="Optional IC supplying MW BH/source identity metadata.")
    parser.add_argument("--apec-data-dir", type=Path, default=ROOT / "erosita_xray",
                        help="Directory containing apec_emissivity_v3.h5.")
    parser.add_argument("--buff-size", type=int, default=1200)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--time-origin-code", type=float,
                        help="Optional code-time origin for an elapsed-Myr label.")
    parser.add_argument("--temperature-mode", choices=("slice", "slab"))
    parser.add_argument("--slab-half-depth", type=float)
    parser.add_argument("--temperature-zlim", type=float, nargs=2)
    parser.add_argument("--density-color-scale", choices=("auto", "fixed"))
    parser.add_argument("--density-zlim", type=float, nargs=2)
    parser.add_argument("--xray-color-scale", choices=("auto", "fixed"))
    parser.add_argument("--xray-zlim", type=float, nargs=2)
    parser.add_argument("--make-video", action="store_true",
                        help="Use the original script's ffmpeg encoder.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--video-name", default="morphology_v10_log.mp4")
    parser.epilog = (
        "Defaults preserve v10_log: x view (y-z image), no disk masking, "
        "temperature slice with fixed 1e5-3e8 K scale, automatic density/X-ray "
        "scales, 0.6-1.0 keV full-depth X-ray projection. Supports single-file "
        "snapshot_NNN.hdf5 only. Existing plot files are never overwritten."
    )
    return parser


def stat_record(path):
    info = path.stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_new_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_snapshots(args, parser):
    directory = args.snapshot_dir.expanduser().resolve(strict=True)
    if not directory.is_dir():
        parser.error("--snapshot-dir must be a directory")
    indexed = []
    for path in directory.iterdir():
        match = PATTERN.fullmatch(path.name)
        if match and path.is_file():
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if len({number for number, _ in indexed}) != len(indexed):
        parser.error("Ambiguous duplicate snapshot numbers in the input directory")
    if args.indices is not None:
        requested = set(args.indices)
        absent = requested - {number for number, _ in indexed}
        if absent:
            parser.error("Requested snapshots are absent: " + str(sorted(absent)))
        indexed = [(number, path) for number, path in indexed if number in requested]
    else:
        indexed = [(number, path) for number, path in indexed
                   if (args.start is None or number >= args.start)
                   and (args.stop is None or number <= args.stop)]
        indexed = indexed[::args.stride]
    if not indexed:
        parser.error("No single-file snapshot_NNN.hdf5 selected; split snapshots are not supported by this wrapper")
    return indexed


def source_module():
    if not SOURCE.is_file():
        raise FileNotFoundError("Keep the repository layout intact: " + str(SOURCE))
    spec = importlib.util.spec_from_file_location("erosita_original_v10_log", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_arguments(module, args):
    values = ["--buff-size", str(args.buff_size), "--dpi", str(args.dpi),
              "--apec-data-dir", str(args.apec_data_dir.expanduser().resolve()),
              "--disk-mask-half-height", "0", "--fps", str(args.fps)]
    for name in ("temperature_mode", "slab_half_depth", "temperature_zlim",
                 "density_color_scale", "density_zlim",
                 "xray_color_scale", "xray_zlim"):
        value = getattr(args, name)
        if value is not None:
            values.append("--" + name.replace("_", "-"))
            values.extend(map(str, value if isinstance(value, list) else [value]))
    parser = module.build_parser()
    result = parser.parse_args(values)
    for fov in args.fovs:
        result.fov = fov
        module.validate_arguments(parser, result)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride < 1 or args.fps < 1 or args.dpi < 1:
        parser.error("--stride, --fps and --dpi must be positive")
    if args.indices is not None and (args.stop is not None or args.stride != 1):
        parser.error("--indices cannot be combined with --stop or a non-default --stride")
    if args.start is not None and args.stop is not None and args.start > args.stop:
        parser.error("--start must not exceed --stop")
    if Path(args.video_name).name != args.video_name:
        parser.error("--video-name must be a filename, not a path")
    selected = select_snapshots(args, parser)
    args.fovs = list(dict.fromkeys(args.fovs))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A single run owns this manifest and its products. Use a new output
    # directory for another selection, so incomplete runs remain reviewable.
    manifest_path = output / "render_manifest.json"
    if manifest_path.exists():
        parser.error("Output already has a render manifest; choose a new output directory")
    for fov in args.fovs:
        for number, _ in selected:
            for field in ("density", "temperature", "xray"):
                if (output / ("fov" + str(fov)) / (field + "_snapshot_" + format(number, "03d") + ".png")).exists():
                    parser.error("Existing PNG would be overwritten; choose a new output directory")
    os.environ["MPLCONFIGDIR"] = str(output / ".mpl-cache")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT))
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gizmo_compat import load_gizmo

    module = source_module()
    plot_args = original_arguments(module, args)
    input_state = {str(path): stat_record(path) for _, path in selected}
    reference_metadata = {}
    if args.reference_ic is not None:
        reference = args.reference_ic.expanduser().resolve(strict=True)
        with h5py.File(reference, "r") as handle:
            reference_metadata.update(module.collect_metadata(handle))
        input_state[str(reference)] = stat_record(reference)
    manifest = {
        "schema": "portable-v10-log-render-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
        "source_script": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "source_function_policy": "Unmodified v10_log make_*_plot, save_single_yt_plot and make_video",
        "selected_indices": [number for number, _ in selected],
        "fov_full_width_kpc": args.fovs,
        "input_stats_before": input_state,
        "axis": "x",
        "image_axes": ["y", "z"],
        "disk_mask_half_height_kpc": 0,
        "temperature_mode": plot_args.temperature_mode,
        "temperature_limits_K": list(plot_args.temperature_zlim),
        "density_color_scale": plot_args.density_color_scale,
        "xray_color_scale": plot_args.xray_color_scale,
        "xray_definition": "0.6-1.0 keV full-depth emissivity integral; unabsorbed; no 1/(4*pi)",
        "cache_policy": "Native kernel fields required; temporary yt cache stays beneath output directory",
        "frames": [],
        "videos": [],
    }
    save_new_json(manifest_path, manifest)
    rendered = {fov: {label: [] for label in ("density", "temperature", "xray")}
                for fov in args.fovs}
    try:
        for number, snapshot in selected:
            if stat_record(snapshot) != input_state[str(snapshot)]:
                raise RuntimeError("Input changed since selection: " + str(snapshot))
            metadata = dict(reference_metadata)
            with h5py.File(snapshot, "r") as handle:
                if int(handle["Header"].attrs.get("NumFilesPerSnapshot", 1)) != 1:
                    raise ValueError("This wrapper requires a single-file snapshot: " + str(snapshot))
                metadata.update(module.collect_metadata(handle))
                time_code = float(handle["Header"].attrs["Time"])
            with tempfile.TemporaryDirectory(prefix=".yt-frame-" + str(number) + "-", dir=output) as temporary:
                dataset = load_gizmo(snapshot, cache_dir=Path(temporary) / "yt_cache",
                                     allow_reconstruction=False)
                center, center_method, _ = module.find_mw_center(dataset, snapshot, metadata)
                module.ensure_kelvin_temperature_field(dataset)
                module.add_apec_metallicity_field(dataset)
                absolute_gyr = float(dataset.current_time.to_value("Gyr"))
                elapsed_myr = None
                if args.time_origin_code is not None:
                    origin_gyr = float(dataset.quan(args.time_origin_code, "code_time").to_value("Gyr"))
                    elapsed_myr = (absolute_gyr - origin_gyr) * 1000.0
                for fov in args.fovs:
                    plot_args.fov = fov
                    directory = output / ("fov" + str(fov))
                    directory.mkdir(exist_ok=True)
                    label = "\nAbsolute t = " + format(absolute_gyr, ".6f") + " Gyr"
                    if elapsed_myr is not None:
                        label += "; elapsed = " + format(elapsed_myr, ".3f") + " Myr"
                    label += "\nFull width " + str(fov) + " kpc"
                    products = []
                    for kind, function in (("density", module.make_density_plot),
                                           ("temperature", module.make_temperature_plot),
                                           ("xray", module.make_xray_plot)):
                        print("RENDER", snapshot.name, fov, kind, flush=True)
                        if kind == "temperature":
                            result = function(dataset, center, label, plot_args, False)
                        else:
                            result = function(dataset, center, label, plot_args)
                        plot = result[0]
                        destination = directory / (kind + "_snapshot_" + format(number, "03d") + ".png")
                        saved = Path(module.save_single_yt_plot(plot, destination, args.dpi))
                        rendered[fov][kind].append(saved)
                        products.append({"field": kind, "file": str(saved.relative_to(output))})
                        del result, plot
                        plt.close("all")
                        gc.collect()
                    manifest["frames"].append({
                        "snapshot": str(snapshot), "index": number, "time_code": time_code,
                        "absolute_time_Gyr": absolute_gyr, "elapsed_Myr": elapsed_myr,
                        "fov_kpc": fov, "center_kpc": center.to_value("kpc").tolist(),
                        "center_method": center_method, "products": products,
                    })
                del dataset
                gc.collect()
            if stat_record(snapshot) != input_state[str(snapshot)]:
                raise RuntimeError("Snapshot changed during plotting: " + str(snapshot))
        if args.make_video:
            base = Path(args.video_name)
            stem, suffix = base.stem, base.suffix or ".mp4"
            for fov in args.fovs:
                directory = output / ("fov" + str(fov))
                for kind, frames in rendered[fov].items():
                    name = kind + "_" + stem + suffix
                    module.make_video(frames, directory, args.fps, name)
                    manifest["videos"].append({
                        "file": str((directory / name).relative_to(output)),
                        "created": (directory / name).is_file(),
                    })
        for name, before in input_state.items():
            if stat_record(Path(name)) != before:
                raise RuntimeError("Input changed: " + name)
        manifest["state"] = "complete"
        manifest["input_stats_unchanged"] = True
    except BaseException as error:
        manifest["state"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
