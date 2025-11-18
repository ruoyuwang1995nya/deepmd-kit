from __future__ import annotations

import numpy as np
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

from deepmd.calculator_pot import DP_POT


def main():
    # Load initial structure
    atoms = read('../data/test.xyz', index=0)

    # Load user potential (scalar) used in the custom energy functional
    potential = -2. #float(np.loadtxt('../data/UPZC_test.txt')[0])

    # Build DeePMD-based calculator (make sure model.ckpt.pt exists here)
    calc = DP_POT(
        model='./model.ckpt.pt',
        head='energy',
        head_potential='potential',
        head_capacity='capacity',
        potential=potential,
    )
    atoms.calc = calc

    # Initialize velocities at 300 K and remove net translation/rotation
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    Stationary(atoms)
    ZeroRotation(atoms)

    # Set up NVT (Langevin) dynamics
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
    log_file = f"nvt_{ts}.log"
    traj_file = f"nvt_{ts}.traj"

    # Logging and trajectory
    logger = MDLogger(dyn, atoms, log_file, header=True, stress=False, peratom=False)
    dyn.attach(logger, interval=10)
    traj = Trajectory(traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=50)

    # Run dynamics
    nsteps = 10000
    dyn.run(nsteps)

    print('NVT run finished. Wrote nvt.traj and nvt.log')


if __name__ == '__main__':
    main()
