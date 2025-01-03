# tensor-equiv

Experiment using tensor networks to check quantum circuits for equivalence.

Uses [GeneralBraOpKet](https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/modules/general_state.html#pytket.extensions.cutensornet.general_state.GeneralBraOpKet) from the [pytket-cutensornet](https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/) library. At the moment executing the `main.py` script requires GPUs.

* Check unitary equivalence between two unitary circuits with the same number of qubits. Use `check_equivalence` function.
* Check equivalence between a unitary circuit $A$ with $n$ qubits and a unitary circuit $B$ with $n+k$ qubits. Use the `check_equivalence_with_ancillas` function.