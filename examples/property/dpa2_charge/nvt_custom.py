from __future__ import annotations

from datetime import datetime

from ase import units
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md import MDLogger

from deepmd.calculator_custom import DP_Custom


def main():
    # Load initial structure
    atoms = read('./test.xyz', index=0)

    # Applied external Fermi energy (eV)
    e_fermi = -4.5

    # Build calculator with three model heads: energy, capacity, fermi
    calc = DP_Custom(
        model='./model.ckpt.pt',
        e_fermi=e_fermi,
        head_energy='ener',
        head_capacity='cap',
        head_fermi='fermi',
    )
    atoms.calc = calc

    # Quick sanity check: print initial energy and induced charge
    print(f"Initial energy : {atoms.get_potential_energy():.6f} eV")
    print(f"Induced charge : {calc.results['charge']:.6f} e")

    # Initialize velocities at 300 K and remove net translation/rotation
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    Stationary(atoms)
    ZeroRotation(atoms)

    # NVT (Langevin) dynamics
    timestep = 0.5 * units.fs
    friction = 0.02  # 1/fs
    dyn = Langevin(
        atoms,
        timestep,
        temperature_K=300,
        friction=friction,
        fixcm=True,
    )

    # Timestamped filenames
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"nvt_custom_{ts}.log"
    traj_file = f"nvt_custom_{ts}.traj"

    logger = MDLogger(dyn, atoms, log_file, header=True, stress=False, peratom=False)
    dyn.attach(logger, interval=10)
    traj = Trajectory(traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=50)

    nsteps = 1000
    dyn.run(nsteps)

    print(f"NVT run finished. Wrote {traj_file} and {log_file}")

if __name__ == '__main__':
    main()
