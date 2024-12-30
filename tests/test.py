from pytket.circuit.display import view_browser as draw

from pytket._tket.circuit import Circuit
from pytket.passes import ComposePhasePolyBoxes
from gadgetisation import REPLACE_HADAMARDS


def build_qft_circuit(n_qubits: int) -> Circuit:
    circ = Circuit(n_qubits, name="QFT")
    for i in range(n_qubits):
        circ.H(i)
        for j in range(i + 1, n_qubits):
            circ.CU1(1 / 2 ** (j - i), j, i)
    for k in range(0, n_qubits // 2):
        circ.SWAP(k, n_qubits - k - 1)
    return circ


test_circ = build_qft_circuit(4)

draw(test_circ)
ComposePhasePolyBoxes().apply(test_circ)
draw(test_circ)
REPLACE_HADAMARDS.apply(test_circ)

draw(test_circ)
