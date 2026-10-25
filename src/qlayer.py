"""Simulator-only variational quantum circuit (PennyLane default.qubit, CPU)."""
import pennylane as qml, torch

def vqc_layer(n_qubits=4, n_layers=2):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for w in range(n_qubits): qml.RY(inputs[..., w], wires=w)          # angle encoding
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(weights[l, w, 0], wires=w); qml.RZ(weights[l, w, 1], wires=w)
            for w in range(n_qubits): qml.CNOT(wires=[w, (w + 1) % n_qubits])  # circular entanglement
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 2)})

class QuantumBlock(torch.nn.Module):
    """Linear -> tanh*pi/2 (angles) -> VQC -> <Z> vector."""
    def __init__(self, d_in, n_qubits=4, n_layers=2):
        super().__init__(); self.pre = torch.nn.Linear(d_in, n_qubits); self.q = vqc_layer(n_qubits, n_layers)
    def forward(self, x): return self.q(torch.tanh(self.pre(x)) * torch.pi / 2)
