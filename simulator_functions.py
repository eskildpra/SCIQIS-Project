import numpy as np

from numpy import size


def string_to_state(s):
    """
    Convert string to state, takes 0,1,+,-
    """
    states = []
    for i in s:
        if i == "0":
            states.append(np.array([[1], [0]]))
        elif i == "1":
            states.append(np.array([[0], [1]]))
        elif i == "+":
            states.append(np.array([[1], [1]]) / np.sqrt(2))
        elif i == "-":
            states.append(np.array([[1], [-1]]) / np.sqrt(2))
    for state in states:
        if 'result' in locals():
            result = np.kron(result, state)
        else:
            result = state
    return result
def Hgate(n):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    step = H
    for i in range(n-1):
        step = np.kron(step, H)
    return step

def Rx(theta, size):
    gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
    step = gate
    for i in range(size-1):
        step = np.kron(step, gate)
    return step

def Ry(theta, size):
    gate = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    step = gate
    for i in range(size-1):
        step = np.kron(step, gate)
    return step

def Rz(theta, size):
    gate = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])
    step = gate
    for i in range(size-1):
        step = np.kron(step, gate)
    return step

def H_target(state, target, size):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)
    factors = [H if i == target else I for i in range(size)]
    gate = factors[0]
    for i in range(1, size):
        gate = np.kron(gate, factors[i])
    return gate @ state

def X_target(state, target, size):
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    factors = [X if i == target else I for i in range(size)]
    gate = factors[0]
    for i in range(1, size):
        gate = np.kron(gate, factors[i])
    return gate @ state

def Z_target(state, target, size):
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    factors = [Z if i == target else I for i in range(size)]
    gate = factors[0]
    for i in range(1, size):
        gate = np.kron(gate, factors[i])
    return gate @ state

def Rx_target(state, theta, target, size):
    gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
    I = np.eye(2)
    factors = [gate if i == target else I for i in range(size)]
    gate = factors[0]
    for i in range(1, size):
        gate = np.kron(gate, factors[i])
    return gate @ state

def Ry_target(state, theta, target, size):
    gate = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    I = np.eye(2)
    factors = [gate if i == target else I for i in range(size)]
    gate = factors[0]
    for i in range(1, size):
        gate = np.kron(gate, factors[i])
    return gate @ state

def Rz_target(state, theta, target, size):
    gate = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])
    I = np.eye(2)
    factors = [gate if i == target else I for i in range(size)]
    gate = factors[0]
    for i in range(1, size):
        gate = np.kron(gate, factors[i])
    return gate @ state

def CNOT(control, target, size):
    """
    (|0><0|)_c ⊗ I_rest + (|1><1|)_c ⊗ X_t ⊗ I_rest
    """
    P0 = np.array([[1, 0], [0, 0]]) #checks if qbit is 0, if yes, keep
    P1 = np.array([[0, 0], [0, 1]]) #checks if qbit is 1, if yes, apply X to target
    I  = np.eye(2)
    X  = np.array([[0, 1], [1, 0]])
    # Term 1: Control is |0> -> apply Identity to target
    term0 = [P0 if i == control else I for i in range(size)]
    
    # Term 2: Control is |1> -> apply X to target
    term1 = [X if i == target else (P1 if i == control else I) for i in range(size)]

    #make tensor
    op0 = term0[0]
    op1 = term1[0]
    for i in range(1, size):
        op0 = np.kron(op0, term0[i])
        op1 = np.kron(op1, term1[i])
    return op0 + op1

def CZ(control, target, size):
    """
    (|0><0|)_c ⊗ I_rest + (|1><1|)_c ⊗ Z_t ⊗ I_rest
    """
    P0 = np.array([[1, 0], [0, 0]]) #checks if qbit is 0, if yes, keep
    P1 = np.array([[0, 0], [0, 1]]) #checks if qbit is 1, if yes, apply Z to target
    I  = np.eye(2)
    Z  = np.array([[1, 0], [0, -1]])
    #control is |0> -> apply Identity to target
    term0 = [P0 if i == control else I for i in range(size)]
    # Control is |1> -> apply Z to target
    term1 = [Z if i == target else (P1 if i == control else I) for i in range(size)]
    #make tensor
    op0 = term0[0]
    op1 = term1[0]
    for i in range(1, size):
        op0 = np.kron(op0, term0[i])
        op1 = np.kron(op1, term1[i])
    return op0 + op1

def measurement(state, qbit):
    size = int(np.log2(state.shape[0]))
    #extract the state of the qubit to be measured
    P0 = np.array([[1, 0], [0, 0]]) # |0><0|
    P1 = np.array([[0, 0], [0, 1]]) # |1><1|
    I  = np.eye(2)
    term0= [P0 if i == qbit else I for i in range(size)]
    term1= [P1 if i == qbit else I for i in range(size)]
    #make to tensor 
    op0 = term0[0]
    op1 = term1[0]
    for i in range(1, size):
        op0 = np.kron(op0, term0[i])
        op1 = np.kron(op1, term1[i])
    #probabilities
    p0 = np.real(np.conj(state.T) @ op0 @ state)[0, 0] # prob of measuring |0>
    p1 = np.real(np.conj(state.T) @ op1 @ state)[0, 0] # prob of measuring |1>
    #randomly choose outcome
    outcome = np.random.choice([0, 1], p=[p0, p1])
    #collapse state
    if outcome == 0:
        new_state = op0 @ state / np.sqrt(p0)
    else:
        new_state = op1 @ state / np.sqrt(p1)
    return outcome, new_state, [p0, p1]

def density_matrix(state):
    rho = state @ np.conj(state.T)
    return rho

def trace_out(rho, qubit, size):
    """
    Trace out a qubit from a density matrix. CHATTET
    """
    # Reshape the density matrix to separate the qubit to be traced out
    rho_reshaped = rho.reshape([2] * (2 * size))
    # Move the qubit to be traced out to the last position
    axes = list(range(2 * size))
    axes.remove(qubit)
    axes.append(qubit)
    rho_permuted = np.transpose(rho_reshaped, axes)
    # Reshape to combine the remaining qubits
    new_shape = [2 ** (size - 1), 2 ** (size - 1), 2, 2]
    rho_permuted = rho_permuted.reshape(new_shape)
    # Trace out the last two dimensions (the qubit)
    rho_traced_out = np.trace(rho_permuted, axis1=2, axis2=3)
    return rho_traced_out


