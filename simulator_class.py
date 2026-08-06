import numpy as np
from simulator_functions import *


class QCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros((2**num_qubits, 1), dtype=complex)
        self.state[0, 0] = 1 

    def apply_gate(self, gate, target):
        # Accept either a callable gate function (like H_target)
        # or a full gate matrix (ndarray).
        if callable(gate):
            self.state = gate(self.state, target, self.num_qubits)
        else:
            # assume `gate` is a full operator matrix acting on the whole system
            self.state = gate @ self.state

    def measure(self, qubit):
        outcome, self.state, probabilities = measurement(self.state, qubit)   
        return outcome, probabilities

    def apply_cnot(self, control, target):
        self.state = CNOT(control, target, self.num_qubits) @ self.state

    def bell_state(self, qubit1, qubit2):
        self.apply_gate(H_target, qubit1)
        self.apply_cnot(qubit1, qubit2)

    def rotation(self, theta, axis, target):
        if axis == 'x':
            self.state = Rx_target(self.state, theta, target, self.num_qubits)
        elif axis == 'y':
            self.state = Ry_target(self.state, theta, target, self.num_qubits)
        elif axis == 'z':
            self.state = Rz_target(self.state, theta, target, self.num_qubits)


class RandomQCircuit(QCircuit):
    def __init__(self, num_qubits, depth):
        super().__init__(num_qubits)
        self.depth = depth
        gatelist = ["H", "X", "Y", "Z", "Rx", "Ry", "Rz", "CNOT"]
        for qbit in range(self.num_qubits):
            for d in range(self.depth):
                gate = np.random.choice(gatelist)
                if gate in ["H", "X", "Y", "Z"]:
                    self.apply_gate(globals()[f"{gate}_target"], qbit)
                elif gate in ["Rx", "Ry", "Rz"]:
                    theta = np.random.uniform(0, 2 * np.pi)
                    self.rotation(theta, gate[-1].lower(), qbit)
                elif gate == "CNOT":
                    control = np.random.randint(0, self.num_qubits)
                    target = np.random.randint(0, self.num_qubits)
                    while target == control:
                        target = np.random.randint(0, self.num_qubits)
                    self.apply_cnot(control, target)



