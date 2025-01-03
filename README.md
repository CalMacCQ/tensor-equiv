# tensor-equiv

Experiment using tensor networks to check quantum circuits for equivalence.

Uses [GeneralBraOpKet](https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/modules/general_state.html#pytket.extensions.cutensornet.general_state.GeneralBraOpKet) from the [pytket-cutensornet](https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/) library. At the moment executing the `main.py` script requires GPUs.

* Check unitary equivalence between two unitary circuits with the same number of qubits. Use `check_equivalence` function.
* Check equivalence between a unitary circuit $A$ with $n$ qubits and a unitary circuit $B$ with $n+k$ qubits. Use the `check_equivalence_with_ancillas` function.

## Try it out (requires installation from source)

1. First clone
```shell
git clone git@github.com:CalMacCQ/tensor-equiv.git
```

2. Now install dependencies with [uv](https://docs.astral.sh/uv/).

```shell
cd tensor-equiv
uv sync
```

3. Execute `main` script as a demo (requires GPU).

```shell
cd src
uv run python main.py
```
