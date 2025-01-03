from topt_proto.gadgetisation import (
    REPLACE_HADAMARDS,
    REPLACE_CONDITIONALS,
)
from pytket.circuit import Circuit
from pytket.passes import ComposePhasePolyBoxes

from .checkers import get_ancilla_check_circuit, check_equivalence_with_ancillas

import time


def build_qft_circuit(n_qubits: int) -> Circuit:
    circ = Circuit(n_qubits, name="$$QFT$$")
    for i in range(n_qubits):
        circ.H(i)
        for j in range(i + 1, n_qubits):
            circ.CU1(1 / 2 ** (j - i), j, i)
    for k in range(0, n_qubits // 2):
        circ.SWAP(k, n_qubits - k - 1)
    return circ


def main():
    N_QUBITS = 15
    print(
        f"Checking partial equivalence of two QFT circuits for n ={N_QUBITS}, k ={N_QUBITS - 1}."
    )
    test_circ = build_qft_circuit(n_qubits=N_QUBITS)

    qft = test_circ.copy()

    ComposePhasePolyBoxes().apply(test_circ)

    REPLACE_HADAMARDS.apply(test_circ)

    REPLACE_CONDITIONALS.apply(test_circ)

    test_circ.remove_blank_wires()

    lhs_circ = get_ancilla_check_circuit(qft, test_circ, lhs_circ=True)
    rhs_circ = get_ancilla_check_circuit(qft, test_circ, lhs_circ=False)
    start = time.time()
    print(check_equivalence_with_ancillas(qft, test_circ))
    end = time.time()
    print(f"Time taken: {end - start}s")


if __name__ == "__main__":
    main()
