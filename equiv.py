from pytket.circuit import Circuit, DiagonalBox
from pytket.extensions.cutensornet.general_state import GeneralBraOpKet
from pytket.passes import DecomposeBoxes
import numpy as np


def get_choi_state_circuit(unitary_circ: Circuit) -> Circuit:
    """
    Returns a circuit to prepare a Choi state |phi_U> given a circuit
    implementing U. That is a 2n qubit statevector whose amplitudes
    encode the entries of an n qubit unitary matrix.
    """
    choi_circ = Circuit()

    control_reg = choi_circ.add_q_register("C", unitary_circ.n_qubits)
    target_reg = choi_circ.add_q_register("T", unitary_circ.n_qubits)

    for qubit in control_reg:
        choi_circ.H(qubit)

    for control, target in zip(control_reg, target_reg):
        choi_circ.CX(control, target)

    choi_circ.add_circuit(unitary_circ, list(target_reg))
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
    DecomposeBoxes().apply(circ)
    return circ


def check_equivalence(circuit_a: Circuit, circuit_b: Circuit) -> bool:
    """
    Checks unitary equivalence between two circuits C_A and C_B up to a global phase
    by evaluating the inner product <C_A'|C_B'> using the GeneralBraOpKet class.

    Note: C_U' is the circuit preparing the choi state |phi_U>.
    """

    lhs_circ: Circuit = get_choi_state_circuit(circuit_a)
    rhs_circ: Circuit = get_choi_state_circuit(circuit_b)

    with GeneralBraOpKet(bra=lhs_circ, ket=rhs_circ) as prod:
        overlap = prod.contract()

    return np.isclose(overlap, 1)


def get_ancilla_check_circuit(
    circuit_1: Circuit,
    circuit_2: Circuit,
) -> Circuit:
    assert circuit_1.n_qubits != circuit_2.n_qubits

    circ = circuit_1.dagger()
    circ.append(circuit_2)

    return get_choi_state_circuit(circ)


def _reorder_registers(circ: Circuit) -> Circuit:
    """
    Hacky function to swap two registers by relabelling
    """
    registers = circ.q_registers()
    assert len(registers) == 2
    registers[0].name = "z"
    return circ


def check_equivalence_with_ancillas(circ_a: Circuit, circ_b: Circuit) -> bool:
    assert circ_b.n_qubits > circ_a.n_qubits

    bra_circ = get_ancilla_check_circuit(circ_a.dagger(), circ_b)

    circ_b_prime = _reorder_registers(circ_b.dagger())

    ket_circ = get_ancilla_check_circuit(circ_b_prime, circ_a)

    with GeneralBraOpKet(bra=bra_circ, ket=ket_circ) as prod:
        overlap = prod.contract()

    return np.isclose(overlap, 1)
