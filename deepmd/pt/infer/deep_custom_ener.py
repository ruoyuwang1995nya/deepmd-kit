# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference class for ener_intensive (CustomIntensiveEnergyModel) heads."""

from deepmd.infer.deep_property import DeepProperty

__all__ = ["DeepCustomEner"]


class DeepCustomEner(DeepProperty):
    """Inference for ener_intensive model heads (CustomIntensiveEnergyModel).

    ``DeepProperty`` cannot be used directly for ``ener_intensive`` heads
    because its ``change_output_def()`` calls ``get_task_dim()`` and
    ``get_intensive()`` on the JIT-compiled model, but
    ``CustomIntensiveEnergyModel`` does not export those methods (they belong
    to ``PropertyModel``).

    This class overrides both to hardcoded values that are always correct for
    an intensive scalar property:

    * ``get_task_dim()`` → 1   (scalar output)
    * ``get_intensive()`` → True

    Everything else — ``change_output_def()``, ``eval()``, ``get_var_name()``,
    etc. — is inherited unchanged from ``DeepProperty``.

    Usage
    -----
    >>> from deepmd.pt.infer.deep_custom_ener import DeepCustomEner
    >>> dp = DeepCustomEner("model.ckpt.pt", head="fermi")
    >>> prop, derv = dp.eval(coords, cells, atom_types)
    >>> # prop  shape: (nframes, 1)
    >>> # derv  shape: (nframes, natoms, 1, 3)  — force-convention derivative
    """

    def get_task_dim(self) -> int:
        """Output dimension of an intensive scalar property is always 1."""
        return 1

    def get_intensive(self) -> bool:
        """Intensive property: per-atom outputs are averaged, not summed."""
        return True
