import numpy as np
from pytket import Circuit
from pytket.passes import DecomposeBoxes, RemoveBarriers
from pytket.extensions.cutensornet.general_state import GeneralBraOpKet

from .builders import get_choi_state_circuit, get_ancilla_check_circuit


def check_equivalence(circuit_a: Circuit, circuit_b: Circuit) -> bool:
    """
    Checks unitary equivalence between two circuits C_A and C_B up to a global phase
    by evaluating the inner product <C_A'|C_B'> using the GeneralBraOpKet class.

    Note: C_U' is the circuit preparing the choi state |phi_U>.
    """
    assert circuit_a.n_qubits == circuit_b.n_qubits

    lhs_circ: Circuit = get_choi_state_circuit(circuit_a)
    rhs_circ: Circuit = get_choi_state_circuit(circuit_b)

    with GeneralBraOpKet(bra=lhs_circ, ket=rhs_circ) as prod:
        overlap = prod.contract()

    return np.isclose(overlap, 1)


def check_equivalence_with_ancillas(circuit_a: Circuit, circuit_b: Circuit) -> bool:
    assert circuit_a.n_qubits < circuit_b.n_qubits

    lhs_circ: Circuit = get_ancilla_check_circuit(circuit_a, circuit_b, lhs_circ=True)
    rhs_circ: Circuit = get_ancilla_check_circuit(circuit_a, circuit_b, lhs_circ=False)

    # Preprocessing passes
    DecomposeBoxes().apply(lhs_circ)
    RemoveBarriers().apply(lhs_circ)

    DecomposeBoxes().apply(rhs_circ)
    RemoveBarriers().apply(lhs_circ)

    with GeneralBraOpKet(bra=lhs_circ, ket=rhs_circ) as prod:
        overlap = prod.contract()

    return np.isclose(overlap, 1)
