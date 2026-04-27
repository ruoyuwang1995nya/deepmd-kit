# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
    Union,
)

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)

import torch


@Fitting.register("ener_intensive")
class IntensiveEnergyFittingNet(EnergyFittingNet):
    """Fitting net for an intensive energy-like property with a user-defined name.

    Like :class:`EnergyFittingNet` but two key differences:

    1. **var_name** is configurable (default ``"energy"``).  In multi-task
       training each intensive-energy head should use a distinct name so the
       label files (``{var_name}.npy``, ``{var_name}_derv_r.npy``) don't
       collide with those of other heads.

    2. The output variable is marked ``intensive=True``, so per-atom outputs
       are **averaged** rather than summed during reduction.  Forces follow as
       ``F_i = -d(E_avg)/dr_i = -(1/N) Σ_j d(e_j)/dr_i`` via autograd.

    Parameters
    ----------
    var_name : str
        Name of the intensive property.  Determines the filenames read from
        the dataset (``{var_name}.npy`` for the scalar label and
        ``{var_name}_derv_r.npy`` for the gradient label).
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        var_name: str = "energy",
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        type_map: Optional[list[str]] = None,
        default_fparam: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        # Call InvarFitting directly to pass the custom var_name.
        # EnergyFittingNet.__init__ hardcodes "energy" so we skip it.
        InvarFitting.__init__(
            self,
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            type_map=type_map,
            default_fparam=default_fparam,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                    intensive=True,  # average over atoms instead of summing
                ),
            ]
        )

    def serialize(self) -> dict:
        data = InvarFitting.serialize(self)
        data["type"] = "ener_intensive"
        # var_name is already included by InvarFitting.serialize()
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "IntensiveEnergyFittingNet":
        data = data.copy()
        from deepmd.utils.version import check_version_compatibility
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        # Keep var_name and dim_out (unlike EnergyFittingNet which pops them)
        return super(EnergyFittingNet, cls).deserialize(data)
