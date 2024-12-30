from topt_proto.gadgetisation import (
    REPLACE_HADAMARDS,
    REPLACE_CONDITIONALS,
)
from pytket.circuit import Circuit
from pytket.passes import ComposePhasePolyBoxes
from pytket.circuit.display import view_browser as draw

from tensor_equiv import get_ancilla_check_circuit, check_equivalence_with_ancillas


def build_qft_circuit(n_qubits: int) -> Circuit:
    circ = Circuit(n_qubits, name="$$QFT$$")
    for i in range(n_qubits):
        circ.H(i)
        for j in range(i + 1, n_qubits):
            circ.CU1(1 / 2 ** (j - i), j, i)
    for k in range(0, n_qubits // 2):
        circ.SWAP(k, n_qubits - k - 1)
    return circ


test_circ = build_qft_circuit(4)

qft4 = test_circ.copy()

ComposePhasePolyBoxes().apply(test_circ)


REPLACE_HADAMARDS.apply(test_circ)

REPLACE_CONDITIONALS.apply(test_circ)

test_circ.remove_blank_wires()

lhs_circ = get_ancilla_check_circuit(qft4, test_circ, lhs_circ=True)
rhs_circ = get_ancilla_check_circuit(qft4, test_circ, lhs_circ=False)


draw(lhs_circ)
draw(rhs_circ)

print(check_equivalence_with_ancillas(qft4, test_circ))
