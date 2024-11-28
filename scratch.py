import numpy as np
from pytket import Circuit
from pytket.circuit import DiagonalBox, CircBox
from pytket.passes import DecomposeBoxes
from pytket.circuit.display import view_browser as draw
from pytket.extensions.cutensornet.general_state import GeneralBraOpKet


def get_n_bell_pairs_circuit(
    n_bell_pairs: int, control_name: str, target_name: str
) -> Circuit:
    bell_circ = Circuit()

    control_reg = bell_circ.add_q_register(control_name, n_bell_pairs)
    target_reg = bell_circ.add_q_register(target_name, n_bell_pairs)

    for qubit in control_reg:
        bell_circ.H(qubit)

    for control, target in zip(control_reg, target_reg):
        bell_circ.CX(control, target)

    return bell_circ


def get_choi_state_circuit(unitary_circ: Circuit) -> Circuit:
    choi_circ = get_n_bell_pairs_circuit(
        unitary_circ.n_qubits, control_name="C", target_name="T"
    )
    target_reg = Circuit.get_q_register("T")

    choi_circ.add_circuit(unitary_circ, list(target_reg))
    return choi_circ


def get_diagonal_choi_state_circuit(diag: DiagonalBox) -> Circuit:
    """
    Small optimisation for circuits which correspond to a diagonal unitary.

    Here the Choi state can be prepared with an n qubit circuit instead of 2n.
    """
    circ = Circuit(diag.n_qubits)

    for qubit in range(diag.n_qubits):
        circ.H(qubit)

    circ.add_gate(diag, list(range(diag.n_qubits)))
    DecomposeBoxes().apply(circ)
    return circ


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


def get_ancilla_lhs_circ(circuit_a: Circuit, circuit_b: Circuit) -> Circuit:
    assert circuit_b.n_qubits > circuit_a.n_qubits
    n_ancillas = circuit_b.n_qubits - circuit_a.n_qubits

    ket_circ = Circuit()
    ancilla_reg_r0 = ket_circ.add_q_register("q_r0_ca", n_ancillas)
    r0_bell_pairs_circ = get_n_bell_pairs_circuit(
        circuit_a.n_qubits, control_name="q_r0_c", target_name="q_r0_t"
    )
    ket_circ.append(r0_bell_pairs_circ)
    target_reg_r0 = ket_circ.get_q_register("q_r0_t")

    a_dg_box = CircBox(circuit_a.dagger())
    a_dg_box.circuit_name = "$$A^{\dagger}$$"
    circuit_b.name = "$$B$$"

    ket_circ.add_circbox_regwise(
        CircBox(circuit_b), [target_reg_r0, ancilla_reg_r0], []
    )
    ket_circ.add_gate(a_dg_box, list(target_reg_r0))

    r1_bell_pairs_circ = get_n_bell_pairs_circuit(
        circuit_a.n_qubits, control_name="q_r1_c", target_name="q_r1_t"
    )
    ket_circ_prime = ket_circ * r1_bell_pairs_circ

    return ket_circ_prime


def get_ancilla_rhs_circ(circuit_a: Circuit, circuit_b: Circuit) -> Circuit:
    assert circuit_b.n_qubits > circuit_a.n_qubits
    n_ancillas = circuit_b.n_qubits - circuit_a.n_qubits

    r0_bell_pairs_circ_rhs = get_n_bell_pairs_circuit(
        circuit_a.n_qubits, control_name="q_r0_c", target_name="q_r0_t"
    )

    bra_circ = Circuit()
    bra_circ_prime = bra_circ * r0_bell_pairs_circ_rhs

    ancilla_reg = bra_circ_prime.add_q_register("q_r0_ca", n_ancillas)

    r1_bell_pairs_circ_rhs = get_n_bell_pairs_circuit(
        circuit_a.n_qubits, control_name="q_r1_c", target_name="q_r1_t"
    )

    bra_circ_prime.append(r1_bell_pairs_circ_rhs)
    target_reg_r1 = bra_circ_prime.get_q_register("q_r1_t")

    b_dg_box = CircBox(circuit_b.dagger())
    b_dg_box.circuit_name = "$$B^{\dagger}$$"
    circuit_a.name = "$$A$$"

    bra_circ_prime.add_circbox_regwise(b_dg_box, [target_reg_r1, ancilla_reg], [])
    bra_circ_prime.add_gate(CircBox(circuit_a), list(target_reg_r1))

    return bra_circ_prime


circ_1 = Circuit(2).H(0).CX(0, 1)

circ_2 = Circuit(3).H(2).CX(1, 0).CX(2, 0)

draw(get_ancilla_rhs_circ(circ_1, circ_2))
