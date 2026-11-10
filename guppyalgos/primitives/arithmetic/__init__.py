"""Quantum arithmetic algorithms."""

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
)

from .adder.adder_ripple_gidney import (
    adder_ripple_gidney_carry_out,
    adder_ripple_gidney_mod,
    adder_ripple_gidney_carry_out_dagger,
    adder_ripple_gidney_mod_dagger,
    cntrl_adder_ripple_gidney_carry_out,
    cntrl_adder_ripple_gidney_mod,
    cntrl_adder_ripple_gidney_carry_out_dagger,
    cntrl_adder_ripple_gidney_mod_dagger,
)
from .adder.adder_ripple_cuccaro import (
    adder_ripple_cuccaro_carry_out,
    adder_ripple_cuccaro_mod,
    adder_ripple_cuccaro_carry_out_dagger,
    adder_ripple_cuccaro_mod_dagger,
    cntrl_adder_ripple_cuccaro_carry_out,
    cntrl_adder_ripple_cuccaro_mod,
    cntrl_adder_ripple_cuccaro_mod_dagger,
    cntrl_adder_ripple_cuccaro_carry_out_dagger,
)
from .exponentiator.exponentiator_ripple_gidney import exponentiator_ripple_gidney_mod
from .multiplier.multiplier_ripple_gidney import (
    cntrl_multiplier_ripple_gidney_mod_in_place,
    cntrl_multiplier_ripple_gidney_mod,
    multiplier_ripple_gidney_mod_in_place,
    multiplier_ripple_gidney_mod,
)
from .subtractor.subtractors import (
    subtractor_ripple_gidney_carry_out,
    subtractor_ripple_gidney_mod,
    subtractor_ripple_cuccaro_carry_out,
    subtractor_ripple_cuccaro_mod,
    cntrl_subtractor_ripple_cuccaro_carry_out,
    cntrl_subtractor_ripple_cuccaro_mod,
    cntrl_subtractor_ripple_gidney_carry_out,
    cntrl_subtractor_ripple_gidney_mod,
)

__all__ = [
    "adder_ripple_cuccaro_carry_out",
    "adder_ripple_cuccaro_carry_out_dagger",
    "adder_ripple_cuccaro_mod",
    "adder_ripple_cuccaro_mod_dagger",
    "adder_ripple_gidney_carry_out",
    "adder_ripple_gidney_carry_out_dagger",
    "adder_ripple_gidney_mod",
    "adder_ripple_gidney_mod_dagger",
    "cntrl_adder_ripple_cuccaro_carry_out",
    "cntrl_adder_ripple_cuccaro_carry_out_dagger",
    "cntrl_adder_ripple_cuccaro_mod",
    "cntrl_adder_ripple_cuccaro_mod_dagger",
    "cntrl_adder_ripple_gidney_carry_out",
    "cntrl_adder_ripple_gidney_carry_out_dagger",
    "cntrl_adder_ripple_gidney_mod",
    "cntrl_adder_ripple_gidney_mod_dagger",
    "cntrl_multiplier_ripple_gidney_mod",
    "cntrl_multiplier_ripple_gidney_mod_in_place",
    "cntrl_subtractor_ripple_cuccaro_carry_out",
    "cntrl_subtractor_ripple_cuccaro_mod",
    "cntrl_subtractor_ripple_gidney_carry_out",
    "cntrl_subtractor_ripple_gidney_mod",
    "exponentiator_ripple_gidney_mod",
    "multiplier_ripple_gidney_mod",
    "multiplier_ripple_gidney_mod_in_place",
    "subtractor_ripple_cuccaro_carry_out",
    "subtractor_ripple_cuccaro_mod",
    "subtractor_ripple_gidney_carry_out",
    "subtractor_ripple_gidney_mod",
    "temp_and",
]
