# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASE calculator interface module."""

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

from deepmd.pt.infer.deep_eval import DeepProperty

if TYPE_CHECKING:
    from ase import (
        Atoms,
    )
    from ase.neighborlist import (
        NeighborList,
    )

__all__ = ["DP"]


class DP_POT(Calculator):
    """Implementation of ASE deepmd calculator for proposed potential dependent energy and force evaluation.

    Implemented properties are `energy`, `forces` and `stress`

    Parameters
    ----------
    model : Union[str, Path]
        path to the model
    label : str, optional
        calculator label, by default "DP"
    type_dict : dict[str, int], optional
        mapping of element types and their numbers, best left None and the calculator
        will infer this information from model, by default None
    neighbor_list : ase.neighborlist.NeighborList, optional
        The neighbor list object. If None, then build the native neighbor list.
    head : Union[str, None], optional
        a specific model branch choosing from pretrained model, by default None

    Examples
    --------
    Compute potential energy

    >>> from ase import Atoms
    >>> from deepmd.tf.calculator import DP
    >>> water = Atoms('H2O',
    >>>             positions=[(0.7601, 1.9270, 1),
    >>>                        (1.9575, 1, 1),
    >>>                        (1., 1., 1.)],
    >>>             cell=[100, 100, 100],
    >>>             calculator=DP(model="frozen_model.pb"))
    >>> print(water.get_potential_energy())
    >>> print(water.get_forces())

    Run BFGS structure optimization

    >>> from ase.optimize import BFGS
    >>> dyn = BFGS(water)
    >>> dyn.run(fmax=1e-6)
    >>> print(water.get_positions())
    """

    name = "DP"
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
        label: str = "DP",
        type_dict: Optional[dict[str, int]] = None,
        neighbor_list: Optional["NeighborList"] = None,
        head: Optional[str] = None,
        head_potential: Optional[str] = None,
        head_capacity: Optional[str] = None,
        potential: float = 0.0,
        **kwargs: Any,
    ) -> None:
        Calculator.__init__(self, label=label, **kwargs)
        self.dp = DeepPot(
            str(Path(model).resolve()),
            neighbor_list=neighbor_list,
            head=head,
        )
        self.dp_potential = DeepProperty(
            str(Path(model).resolve()),
            neighbor_list=neighbor_list,
            head=head_potential,
        )
        self.dp_capacity = DeepProperty(
            str(Path(model).resolve()),
            neighbor_list=neighbor_list,
            head=head_capacity,
        )
        self.potential=potential
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
        """Run calculation with modified deepmd model.

        Parameters
        ----------
        atoms : Optional[Atoms], optional
            atoms object to run the calculation on, by default None
        properties : list[str], optional
            unused, only for function signature compatibility,
            by default ["energy", "forces", "stress"]
        system_changes : list[str], optional
            unused, only for function signature compatibility, by default all_changes
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]

        fparam = self.atoms.info.get("fparam", None)
        aparam = self.atoms.info.get("aparam", None)
        e, f, _ = self.dp.eval(
            coords=coord, cells=cell, atom_types=atype, fparam=fparam, aparam=aparam
        )[:3]
        
        u_pzc,u_pzc_derv=self.dp_potential.eval(
            coords=coord, cells=cell, atom_types=atype, fparam=fparam, aparam=aparam
        )[:2]
        
        c_pzc,c_pzc_derv=self.dp_capacity.eval(
            coords=coord, cells=cell, atom_types=atype, fparam=fparam, aparam=aparam
        )[:2]
        
        e_tot= 0.5*c_pzc[0][0]*(self.potential - u_pzc[0][0])**2  + e[0][0]

        f_tot = f[0]+ 0.5*c_pzc_derv[0].reshape(-1,3)*(self.potential - u_pzc[0][0])**2 - c_pzc[0][0]*(self.potential - u_pzc[0][0])*u_pzc_derv[0].reshape(-1,3)

        v_tot = np.zeros((3, 3))
        for r, f in zip(coord.reshape(-1, 3), f_tot):
            v_tot -= np.outer(r, f)
            
        self.results["energy"] = e_tot
        self.results["free_energy"] = e_tot
        self.results["forces"] = f_tot
        self.results["virial"] = v_tot#[0].reshape(3, 3)
        
        # convert virial into stress for lattice relaxation
        if cell is not None:
            # the usual convention (tensile stress is positive)
            # stress = -virial / volume
            stress = -0.5 * (v_tot.copy() + v_tot.copy().T) / atoms.get_volume()
            #stress = -v_tot / atoms.get_volume()
            # Voigt notation
            self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        elif "stress" in properties:
            raise PropertyNotImplementedError
        else:
            pass


   
