#!/usr/bin/env python3
"""Convert raw data in .tests/barder/newalldata to DeePMD-kit dataset format.

Writes box, coord, type info, standard energy/force, and all intensive
properties into a single output directory so one dataset can serve both
a standard energy head and multiple ener_intensive heads.

Source files (SRC_DIR):
  box.raw          (nframes, 9)
  coord.raw        (nframes, natoms*3)
  type.raw         (natoms,)
  type_map.raw     one element per line
  energy.raw       (nframes,)       -- extensive total energy
  force.raw        (nframes, natoms*3)
  fermi.raw        (nframes,)       -- intensive Fermi energy
  fermi_derv_r.raw (nframes, natoms*3)
  0.5C-1.raw       (nframes,)       -- intensive cap energy
  cap_derv_r.raw   (nframes, natoms*3)

Output layout (DST_DIR):
  type.raw
  type_map.raw
  set.000/
    box.npy
    coord.npy
    energy.npy          <- energy.raw
    force.npy           <- force.raw
    fermi.npy           <- fermi.raw
    fermi_derv_r.npy    <- fermi_derv_r.raw
    cap.npy             <- 0.5C-1.raw
    cap_derv_r.npy      <- cap_derv_r.raw

Frame selection:
  --frames 0:100       first 100 frames (Python slice syntax)
  --frames 500:600     frames 500-599
  --frames ::10        every 10th frame
  Default: all frames.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


# Mapping: output var_name -> (scalar_source_file, derv_source_file)
# Edit this list to add or remove intensive properties.
INTENSIVE_PROPERTIES = [
    ("fermi", "fermi.raw",   "fermi_derv_r.raw"),
    ("cap",   "0.5C-1.raw",  "cap_derv_r.raw"),
]


def parse_slice(s: str) -> slice:
    """Parse a Python slice string like '0:100', '::10', '500:' into a slice."""
    parts = s.split(":")
    if len(parts) == 1:
        idx = int(parts[0])
        return slice(idx, idx + 1)
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop  = int(parts[1]) if parts[1] else None
        return slice(start, stop)
    if len(parts) == 3:
        start = int(parts[0]) if parts[0] else None
        stop  = int(parts[1]) if parts[1] else None
        step  = int(parts[2]) if parts[2] else None
        return slice(start, stop, step)
    raise ValueError(f"Cannot parse slice: {s!r}")


def convert(src_dir: Path, dst_dir: Path, frame_slice: slice) -> None:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    set_dir = dst_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)

    # --- type.raw (not frame-indexed) ---
    type_arr = np.loadtxt(src_dir / "type.raw", dtype=int)
    np.savetxt(dst_dir / "type.raw", type_arr, fmt="%d")
    natoms = type_arr.shape[0]
    print(f"type.raw            : {natoms} atoms")

    # --- type_map.raw ---
    shutil.copy(src_dir / "type_map.raw", dst_dir / "type_map.raw")
    with open(src_dir / "type_map.raw") as f:
        elements = [l.strip() for l in f if l.strip()]
    print(f"type_map.raw        : {elements}")

    def load(filename: str, expected_cols: int | None = None) -> np.ndarray:
        """Load a .raw file and apply the frame slice."""
        data = np.loadtxt(src_dir / filename, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1) if expected_cols == 1 else data
        data = data[frame_slice]
        if expected_cols is not None and data.ndim == 2:
            assert data.shape[1] == expected_cols, (
                f"{filename}: expected {expected_cols} cols, got {data.shape[1]}"
            )
        return data

    def save(arr: np.ndarray, name: str, src_name: str) -> None:
        np.save(set_dir / f"{name}.npy", arr)
        shape_str = str(arr.shape)
        print(f"{name}.npy{'':{18-len(name)}}: {shape_str:<20}  (from {src_name})")

    # --- box ---
    box = load("box.raw", expected_cols=9)
    nframes = box.shape[0]
    save(box, "box", "box.raw")

    # --- coord ---
    coord = load("coord.raw", expected_cols=natoms * 3)
    assert coord.shape[0] == nframes
    save(coord, "coord", "coord.raw")

    # --- standard energy / force ---
    print()
    energy = load("energy.raw")
    energy = energy.reshape(nframes) if energy.ndim == 2 else energy
    assert energy.shape == (nframes,)
    save(energy, "energy", "energy.raw")

    force = load("force.raw", expected_cols=natoms * 3)
    assert force.shape[0] == nframes
    save(force, "force", "force.raw")

    # --- intensive properties ---
    print()
    for var_name, scalar_src, derv_src in INTENSIVE_PROPERTIES:
        scalar = load(scalar_src)
        scalar = scalar.reshape(nframes) if scalar.ndim == 2 else scalar
        assert scalar.shape == (nframes,), (
            f"[{var_name}] scalar shape {scalar.shape} != ({nframes},)"
        )
        save(scalar, var_name, scalar_src)

        derv = load(derv_src, expected_cols=natoms * 3)
        assert derv.shape[0] == nframes
        save(derv, f"{var_name}_derv_r", derv_src)

    print(f"\n{nframes} frames written to: {dst_dir}")
    print("Multi-task config snippet:")
    print('  "ener": {fitting_net: {type: "ener"}, loss: {type: "ener"}}')
    for var_name, _, _ in INTENSIVE_PROPERTIES:
        print(f'  "{var_name}": {{fitting_net: {{type: "ener_intensive", var_name: "{var_name}"}}, loss: {{type: "ener_intensive"}}}}')


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src", type=Path, default=here,
        help="Source directory containing .raw files (default: script directory)",
    )
    parser.add_argument(
        "--dst", type=Path, default=here / "data_multitask",
        help="Output dataset directory (default: <src>/data_multitask)",
    )
    parser.add_argument(
        "--frames", "-f", type=str, default=":",
        metavar="START:STOP[:STEP]",
        help=(
            "Frame selection as a Python slice (default: all frames). "
            "Examples: 0:100  500:600  ::10"
        ),
    )
    args = parser.parse_args()
    convert(args.src, args.dst, parse_slice(args.frames))


if __name__ == "__main__":
    main()
