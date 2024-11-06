from pytket.circuit import Circuit, CircBox, DiagonalBox
from pytket.extensions.cutensornet import GeneralBraOpKet
from typing import Optional
import numpy as np


def get_choi_state_circuit(unitary_circ: Circuit, u_name: Optional[str]) -> Circuit:
    """
    Returns a circuit to prepare a Choi state |phi_U> given a circuit
    implementing U. That is a 2n qubit statevector whose amplitudes
    encode the entries of an n qubit unitary matrix.
    """
    choi_circ = Circuit()

    # Add control register
    control_reg = choi_circ.add_q_register("C", unitary_circ.n_qubits)

    # Add target register
    target_reg = choi_circ.add_q_register("T", unitary_circ.n_qubits)

    for qubit in control_reg:
        choi_circ.H(qubit)

    for control, target in zip(control_reg, target_reg):
        choi_circ.CX(control, target)

    if u_name:
        unitary_circ.name = u_name

    choi_circ.add_gate(CircBox(unitary_circ), list(target_reg))
    return choi_circ


def get_diagonal_choi_state_circuit(diag: DiagonalBox) -> Circuit:
    """
    Small optimisation for circuits which correspond to a diagonal unitary.

    Here the Choi state can be prepared with an n qubit circuit instead of 2n.
    """
    n_qubits = diag.n_qubits
    circ = Circuit(n_qubits)

    for qubit in range(n_qubits):
        circ.H(qubit)

    circ.add_gate(diag, list(range(n_qubits)))
    return circ


def check_equivalence(circuit_a: Circuit, circuit_b: Circuit) -> bool:
    """
    Checks unitary equivalence between two circuits C_A and C_B up to a global phase
    by evaluating the inner product <C_A'|C_B'> using the GeneralBraOpKet class.

    Note: C_U' is the circuit preparing the choi state |phi_U>.
    """

    lhs_circ: Circuit = get_choi_state_circuit(circuit_a, u_name="$$A$$")
    rhs_circ: Circuit = get_choi_state_circuit(circuit_b, u_name="$$B$$")

    with GeneralBraOpKet(bra=lhs_circ, ket=rhs_circ) as prod:
        overlap = prod.contract()

    return np.isclose(overlap, 1)
