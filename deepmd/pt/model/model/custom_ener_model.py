# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPIntensiveEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener_intensive")
class CustomIntensiveEnergyModel(DPModelCommon, DPIntensiveEnergyModel_):
    """Energy model for intensive (per-atom-averaged) energy labels.

    Like :class:`EnergyModel` but pairs with
    :class:`~deepmd.pt.model.task.custom_ener.IntensiveEnergyFittingNet`
    whose ``output_def`` has ``intensive=True``.  Reduction averages per-atom
    energies, and forces are the negative gradient of that mean energy w.r.t.
    atomic coordinates.

    All output keys are derived from the fitting net's ``var_name`` so that
    multiple heads can coexist in a multi-task run without filename collision:

    * ``{var_name}``           — intensive scalar (per frame)
    * ``atom_{var_name}``      — per-atom contributions
    * ``{var_name}_derv_r``    — force-like gradients (shape ``[nframes, natoms, 3]``)
    * ``{var_name}_derv_c_redu`` / ``{var_name}_derv_c`` — virial (optional)

    The corresponding dataset files are ``{var_name}.npy`` and
    ``{var_name}_derv_r.npy``, compatible with :class:`IntensiveEnergyLoss`.

    Usage in ``input.json``::

        "fitting_net": {"type": "ener_intensive", "var_name": "fermi", ...}
        "loss":        {"type": "ener_intensive", ...}
    """

    model_type = "ener_intensive"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPIntensiveEnergyModel_.__init__(self, *args, **kwargs)

    @torch.jit.export
    def get_var_name(self) -> str:
        """Return the property name used for output keys and dataset filenames."""
        return self.get_fitting_net().var_name

    def translated_output_def(self) -> dict[str, Any]:
        vn = self.get_var_name()
        out_def_data = self.model_output_def().get_data()
        output_def: dict[str, Any] = {
            f"atom_{vn}": out_def_data[vn],
            vn: out_def_data[f"{vn}_redu"],
        }
        if self.do_grad_r(vn):
            output_def[f"{vn}_derv_r"] = out_def_data[f"{vn}_derv_r"]
            output_def[f"{vn}_derv_r"].squeeze(-2)
        if self.do_grad_c(vn):
            output_def[f"{vn}_derv_c_redu"] = out_def_data[f"{vn}_derv_c_redu"]
            output_def[f"{vn}_derv_c_redu"].squeeze(-2)
            output_def[f"{vn}_derv_c"] = out_def_data[f"{vn}_derv_c"]
            output_def[f"{vn}_derv_c"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        vn = self.get_var_name()
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict: dict[str, torch.Tensor] = {}
        model_predict[f"atom_{vn}"] = model_ret[vn]
        model_predict[vn] = model_ret[f"{vn}_redu"]
        if self.do_grad_r(vn):
            model_predict[f"{vn}_derv_r"] = model_ret[f"{vn}_derv_r"].squeeze(-2)
        if self.do_grad_c(vn):
            model_predict[f"{vn}_derv_c_redu"] = model_ret[f"{vn}_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict[f"{vn}_derv_c"] = model_ret[f"{vn}_derv_c"].squeeze(-3)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        vn = self.get_var_name()
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        model_predict: dict[str, torch.Tensor] = {}
        model_predict[f"atom_{vn}"] = model_ret[vn]
        model_predict[vn] = model_ret[f"{vn}_redu"]
        if self.do_grad_r(vn):
            model_predict[f"extended_{vn}_derv_r"] = model_ret[f"{vn}_derv_r"].squeeze(-2)
        if self.do_grad_c(vn):
            model_predict[f"{vn}_derv_c_redu"] = model_ret[f"{vn}_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict[f"extended_{vn}_derv_c"] = model_ret[f"{vn}_derv_c"].squeeze(-3)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
