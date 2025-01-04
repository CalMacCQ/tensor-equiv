import pytest

import numpy as np

from pytket.circuit import Circuit, OpType
from pytket.utils import compare_statevectors
from pytket.passes import ComposePhasePolyBoxes

from tensor_equiv.builders import get_choi_state_circuit, get_ancilla_check_circuit
from tensor_equiv.main import build_qft_circuit

from topt_proto.gadgetisation import REPLACE_CONDITIONALS, REPLACE_HADAMARDS


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


n_qubit_cases = [2, 4, 7]


@pytest.mark.parametrize("n", n_qubit_cases)
def test_ancilla_check_circuits(n) -> None:
    test_circ = build_qft_circuit(n_qubits=n)
    qft = test_circ.copy()

    # Rewrite QFT in terms of {PhasePolyBox, H}
    ComposePhasePolyBoxes().apply(test_circ)

    # Replace Hadamards with ancilla gadgets
    REPLACE_HADAMARDS.apply(test_circ)

    # Replace conditional gates with unitary gates
    REPLACE_CONDITIONALS.apply(test_circ)

    lhs_circ = get_ancilla_check_circuit(qft, test_circ, lhs_circ=True)
    rhs_circ = get_ancilla_check_circuit(qft, test_circ, lhs_circ=False)

    assert lhs_circ.n_qubits == rhs_circ.n_qubits == 5 * n - 1
    assert (
        lhs_circ.n_gates_of_type(OpType.CircBox)
        == rhs_circ.n_gates_of_type(OpType.CircBox)
        == 2
    )
    lhs_boxes = lhs_circ.commands_of_type(OpType.CircBox)
    rhs_boxes = rhs_circ.commands_of_type(OpType.CircBox)
    assert lhs_boxes[0].op.circuit_name == "$$B$$"
    assert lhs_boxes[1].op.circuit_name == "$$A^{\dagger}$$"

    assert rhs_boxes[0].op.circuit_name == "$$A^{\dagger}$$"
    assert rhs_boxes[1].op.circuit_name == "$$B$$"
