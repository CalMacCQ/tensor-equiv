from pytket.circuit import Circuit, OpType
from pytket.passes import CustomPass

from topt_proto.utils import initialise_registers


# Pretty hacky function, only replaces conditional X currently.
def replace_conditionals(circ: Circuit) -> Circuit:
    circ_prime = initialise_registers(circ)

    for cmd in circ:
        match cmd.op.type:
            case OpType.Measure:
                control_qubit = cmd.qubits[0]
                continue

            case OpType.Conditional:
                if cmd.op.width != 1:
                    raise NotImplementedError(
                        f"Replacement not implemented for more than one condition bit ({cmd.op} has {cmd.op.width})."
                    )
                if cmd.op.op.type == OpType.X:
                    target_qubit = cmd.qubits[0]
                    circ_prime.CX(control_qubit, target_qubit)
                else:
                    raise NotImplementedError(
                        f"Replacement for {cmd.op.op.type} not implemented."
                    )
            case OpType.Barrier:
                circ_prime.add_barrier(cmd.qubits)
            case _:
                circ_prime.add_gate(cmd.op, cmd.args)

    circ_prime.remove_blank_wires()

    return circ_prime


REPLACE_CONDITIONALS = CustomPass(replace_conditionals)
