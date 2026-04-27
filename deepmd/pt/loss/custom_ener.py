# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any

import torch

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class IntensiveEnergyLoss(TaskLoss):
    """Loss for an intensive energy-like property with force-like gradients.

    Mirrors :class:`~deepmd.pt.loss.ener.EnergyStdLoss` but reads label data
    from ``{var_name}.npy`` and ``{var_name}_derv_r.npy`` instead of the
    hardcoded ``energy.npy`` / ``force.npy``.  This allows multiple intensive-
    energy heads to coexist in a multi-task run without filename collisions.

    Parameters
    ----------
    var_name : str
        Name of the property.  Must match the ``var_name`` of the corresponding
        :class:`~deepmd.pt.model.task.custom_ener.IntensiveEnergyFittingNet`.
    starter_learning_rate : float
        Learning rate at the start of training (used to compute the prefactor
        schedule).
    start_pref_e, limit_pref_e : float
        Prefactors for the scalar (energy-like) loss at the start and end of
        training.  Set both to ``0`` to disable.
    start_pref_f, limit_pref_f : float
        Prefactors for the gradient (force-like) loss at the start and end of
        training.  Set both to ``0`` to disable.
    """

    def __init__(
        self,
        var_name: str,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 0.0,
        limit_pref_e: float = 0.0,
        start_pref_f: float = 0.0,
        limit_pref_f: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self.starter_learning_rate = starter_learning_rate
        self.has_e = start_pref_e != 0.0 and limit_pref_e != 0.0
        self.has_f = start_pref_f != 0.0 and limit_pref_f != 0.0
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Compute energy and force losses for the intensive property.

        Returns
        -------
        model_pred : dict[str, torch.Tensor]
        loss : torch.Tensor
        more_loss : dict[str, torch.Tensor]
            Contains ``rmse_e`` and/or ``rmse_f`` for logging.
        """
        vn = self.var_name
        vn_derv = f"{vn}_derv_r"

        model_pred = model(**input_dict)

        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss: dict[str, torch.Tensor] = {}
        atom_norm = 1.0 / natoms

        # --- scalar (energy-like) loss ---
        if self.has_e and vn in model_pred and vn in label:
            find_e = label.get(f"find_{vn}", 0.0)
            pref_e = pref_e * find_e
            diff_e = model_pred[vn] - label[vn]
            l2_e = torch.mean(torch.square(diff_e))
            loss += atom_norm * pref_e * l2_e
            more_loss[f"rmse_{vn}"] = self.display_if_exist(
                (l2_e.sqrt() * atom_norm).detach(), find_e
            )
            if mae:
                more_loss[f"mae_{vn}"] = self.display_if_exist(
                    (torch.mean(torch.abs(diff_e)) * atom_norm).detach(), find_e
                )

        # --- gradient (force-like) loss ---
        if self.has_f and vn_derv in model_pred and vn_derv in label:
            find_f = label.get(f"find_{vn_derv}", 0.0)
            pref_f = pref_f * find_f
            diff_f = (label[vn_derv] - model_pred[vn_derv]).reshape(-1)
            l2_f = torch.mean(torch.square(diff_f))
            loss += pref_f * l2_f
            more_loss[f"rmse_{vn}_derv_r"] = self.display_if_exist(
                l2_f.sqrt().detach(), find_f
            )
            if mae:
                more_loss[f"mae_{vn}_derv_r"] = self.display_if_exist(
                    torch.mean(torch.abs(diff_f)).detach(), find_f
                )

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Data files required: ``{var_name}.npy`` and ``{var_name}_derv_r.npy``."""
        req = []
        if self.has_e:
            req.append(
                DataRequirementItem(
                    self.var_name,
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                )
            )
        if self.has_f:
            req.append(
                DataRequirementItem(
                    f"{self.var_name}_derv_r",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return req
