# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASE calculator for charge-corrected energy with three model heads."""

from pathlib import (
    Path,
)
import numpy as np

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Union,
)

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from deepmd.infer import (
    DeepPot,
)

from deepmd.pt.infer.deep_custom_ener import DeepCustomEner

if TYPE_CHECKING:
    from ase import (
        Atoms,
    )
    from ase.neighborlist import (
        NeighborList,
    )

__all__ = ["DP_Custom"]


class DP_Custom(Calculator):
    """ASE calculator for charge-corrected energy using three model heads.

    The system charge `q` is determined by equating the applied external Fermi
    level to the internal Fermi level predicted by the model:

        e_fermi = e_fermi_0 + q / (2 * capacity_0)
        => q = 2 * capacity_0 * (e_fermi - e_fermi_0)

    The total charge-corrected energy is:

        E_tot = e_0 + e_fermi_0 * q + capacity_0 * q^2

    Forces are derived analytically via the chain rule.

    Parameters
    ----------
    model : Union[str, Path]
        Path to the multi-head model checkpoint.
    e_fermi : float
        Applied (external) Fermi energy in the same units as the model.
    label : str, optional
        Calculator label, by default "DP_Custom".
    type_dict : dict[str, int], optional
        Mapping of element symbols to type indices.  If None, inferred from
        the energy head of the model.
    neighbor_list : ase.neighborlist.NeighborList, optional
        Custom neighbor list.  If None, the native DeePMD neighbor list is
        used.
    head_energy : str, optional
        Name of the energy head in the multi-task model.
    head_capacity : str, optional
        Name of the capacity head in the multi-task model.
    head_fermi : str, optional
        Name of the Fermi-level head in the multi-task model.

    Examples
    --------
    >>> from ase import Atoms
    >>> from deepmd.calculator_custom import DP_Custom
    >>> atoms = Atoms(...)
    >>> calc = DP_Custom(
    ...     model="model.pt",
    ...     e_fermi=-4.5,
    ...     head_energy="energy",
    ...     head_capacity="capacity",
    ...     head_fermi="fermi",
    ... )
    >>> atoms.set_calculator(calc)
    >>> print(atoms.get_potential_energy())
    >>> print(atoms.get_forces())
    """

    name = "DP_Custom"
    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "virial",
        "stress",
    ]

    def __init__(
        self,
        model: Union[str, "Path"],
        e_fermi: float,
        label: str = "DP_Custom",
        type_dict: Optional[dict[str, int]] = None,
        neighbor_list: Optional["NeighborList"] = None,
        head_energy: Optional[str] = None,
        head_capacity: Optional[str] = None,
        head_fermi: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        Calculator.__init__(self, label=label, **kwargs)
        model_path = str(Path(model).resolve())
        self.dp = DeepPot(
            model_path,
            neighbor_list=neighbor_list,
            head=head_energy,
        )
        self.dp_capacity = DeepCustomEner(
            model_path,
            neighbor_list=neighbor_list,
            head=head_capacity,
        )
        self.dp_fermi = DeepCustomEner(
            model_path,
            neighbor_list=neighbor_list,
            head=head_fermi,
        )
        self.e_fermi = e_fermi
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
            )

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: list[str] = ["energy", "forces", "virial"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Run the charge-corrected calculation.

        Parameters
        ----------
        atoms : Optional[Atoms], optional
            Atoms object.  If None, uses the previously set atoms.
        properties : list[str], optional
            Unused; present for ASE interface compatibility.
        system_changes : list[str], optional
            Unused; present for ASE interface compatibility.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        coord = self.atoms.get_positions().reshape([1, -1])
        cell = (
            self.atoms.get_cell().reshape([1, -1])
            if sum(self.atoms.get_pbc()) > 0
            else None
        )
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]

        fparam = self.atoms.info.get("fparam", None)
        aparam = self.atoms.info.get("aparam", None)

        # --- evaluate three heads ---
        e_0, f_0, _ = self.dp.eval(
            coords=coord, cells=cell, atom_types=atype, fparam=fparam, aparam=aparam
        )[:3]

        capacity_0, g_C = self.dp_capacity.eval(
            coords=coord, cells=cell, atom_types=atype, fparam=fparam, aparam=aparam
        )[:2]

        e_fermi_0, g_phi = self.dp_fermi.eval(
            coords=coord, cells=cell, atom_types=atype, fparam=fparam, aparam=aparam
        )[:2]

        # --- scalars ---
        phi_0 = float(e_fermi_0[0][0])   # predicted zero-charge Fermi level
        C_0   = float(capacity_0[0][0])  # predicted capacitance-like prefactor
        delta_phi = self.e_fermi - phi_0  # Fermi level offset

        # charge induced by the applied Fermi level:
        #   e_fermi = phi_0 + q / (2 * C_0)  =>  q = 2 * C_0 * delta_phi
        q = 2.0 * C_0 * delta_phi

        # --- total energy ---
        #   E_tot = e_0 + phi_0 * q + C_0 * q^2
        e_tot = float(e_0[0][0]) + phi_0 * q + C_0 * q**2

        # --- total forces (chain rule) ---
        # g_phi[0], g_C[0] are force-convention derivatives, i.e.
        #   g_phi = -d(phi_0)/dr,  g_C = -d(C_0)/dr
        #
        # Derivation:
        #   F_j = -dE_tot/dr_j
        #       = f_0_j
        #         + g_phi_j * [q - 2*C_0*(phi_0 + 2*C_0*q)]
        #         + g_C_j   * [q^2 + 2*(phi_0 + 2*C_0*q)*delta_phi]
        g_phi_flat = g_phi[0].reshape(-1, 3)  # (natoms, 3)
        g_C_flat   = g_C[0].reshape(-1, 3)    # (natoms, 3)

        inner = phi_0 + 2.0 * C_0 * q        # phi_0 + 2*C_0*q
        coef_phi = q - 2.0 * C_0 * inner
        coef_C   = q**2 + 2.0 * inner * delta_phi

        f_tot = f_0[0] + g_phi_flat * coef_phi + g_C_flat * coef_C

        # --- virial (from total forces) ---
        v_tot = np.zeros((3, 3))
        for r, f in zip(coord.reshape(-1, 3), f_tot):
            v_tot -= np.outer(r, f)

        self.results["energy"] = e_tot
        self.results["free_energy"] = e_tot
        self.results["forces"] = f_tot
        self.results["virial"] = v_tot
        self.results["charge"] = q

        if cell is not None:
            stress = -0.5 * (v_tot + v_tot.T) / atoms.get_volume()
            self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        elif "stress" in properties:
            raise PropertyNotImplementedError
