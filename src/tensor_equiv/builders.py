from pytket import Circuit
from pytket.circuit import DiagonalBox, CircBox
from pytket.passes import DecomposeBoxes


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


def get_ancilla_check_circuit(
    circuit_a: Circuit,
    circuit_b: Circuit,
    lhs_circ: bool,
) -> Circuit:

    r0_bell_pairs_circ = get_n_bell_pairs_circuit(
        circuit_a.n_qubits, control_name="q_r0_c", target_name="q_r0_t"
    )

    r1_bell_pairs_circ = get_n_bell_pairs_circuit(
        circuit_a.n_qubits, control_name="q_r1_c", target_name="q_r1_t"
    )

    circ_prime = r0_bell_pairs_circ * r1_bell_pairs_circ

    n_ancillas = circuit_b.n_qubits - circuit_a.n_qubits
    ancilla_reg_r0 = circ_prime.add_q_register("q_r0_ca", n_ancillas)

    a_dg_box = CircBox(circuit_a.dagger())
    a_dg_box.circuit_name = "$$A^{\dagger}$$"
    circuit_b.name = "$$B$$"

    target_reg_r0 = circ_prime.get_q_register("q_r0_t")
    target_reg_r1 = circ_prime.get_q_register("q_r1_t")

    if lhs_circ:
        circ_prime.add_circbox_regwise(
            CircBox(circuit_b), [target_reg_r0, ancilla_reg_r0], []
        )
        circ_prime.add_gate(a_dg_box, list(target_reg_r0))

    else:
        circ_prime.add_gate(a_dg_box, list(target_reg_r1))
        circ_prime.add_circbox_regwise(
            CircBox(circuit_b), [target_reg_r1, ancilla_reg_r0], []
        )

    return circ_prime
