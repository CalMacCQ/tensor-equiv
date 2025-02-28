"""
Demo of partial equivalence checking for QFT circuits with ancillas.
Checks equivalence of a N_QUBIT QFT circuit with a (2*N_QUBIT - 1) implementation
Uses topt-proto to generate ancilla circuits.

Requires GPU for execution.
"""

from topt_proto.gadgetisation import (
    REPLACE_HADAMARDS,
)
from pytket.passes import ComposePhasePolyBoxes

from .builders import build_qft_circuit
from .checkers import check_equivalence_with_ancillas
from .preprocess import REPLACE_CONDITIONALS

import time


def main():
    N_QUBITS = 12
    print(
        f"Checking partial equivalence of two QFT circuits for n ={N_QUBITS}, k ={N_QUBITS - 1}."
    )
    test_circ = build_qft_circuit(n_qubits=N_QUBITS)

    qft = test_circ.copy()

    # Rewrite QFT in terms of {PhasePolyBox, H}
    ComposePhasePolyBoxes().apply(test_circ)

    # Replace Hadamards with ancilla gadgets
    REPLACE_HADAMARDS.apply(test_circ)

    # Replace conditional gates with unitary gates
    REPLACE_CONDITIONALS.apply(test_circ)

    start = time.time()
    print(check_equivalence_with_ancillas(qft, test_circ))
    end = time.time()
    print(f"Time taken: {end - start}s")


if __name__ == "__main__":
    main()
