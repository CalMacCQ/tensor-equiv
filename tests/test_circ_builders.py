import pytest

import numpy as np

from pytket import Circuit
from pytket.utils import compare_statevectors

from tensor_equiv.builders import get_choi_state_circuit, get_ancilla_check_circuit


circuits = [
    Circuit(1).Y(0),
    Circuit(1).H(0),
    Circuit(2).H(0).H(1).CX(0, 1).Rz(0.25, 1).CX(0, 1).H(0).H(1),
]


@pytest.mark.parametrize("circ", circuits)
def test_choi_state_circ(circ) -> None:
    y_vec = circ.get_unitary().reshape(1, 2 ** (2 * circ.n_qubits))
    choi_circ = get_choi_state_circuit(circ)
    assert compare_statevectors(
        y_vec, 1 / (np.sqrt(2**circ.n_qubits)) * choi_circ.get_statevector()
    )


circ_pairs = [
    (
        Circuit(3).CX(0, 1).CX(1, 2).Rz(0.25, 2).CX(1, 2).CX(0, 1),
        Circuit(3).CX(0, 2).CX(1, 2).Rz(0.25, 2).CX(1, 2).CX(0, 2),
    )
]


@pytest.mark.parametrize("circ_pair", circ_pairs)
def test_ancilla_check_circuits(circ_pair) -> None:
    lhs_circ = get_ancilla_check_circuit(circuit_a=circ_pair[0], circuit_b=circ_pair[1])
    assert lhs_circ.n_qubits =
