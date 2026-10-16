"""
autoencoder.py
==============

Step 7: latent-space representation learning.

A compact, fully self-contained autoencoder implemented in pure NumPy so the
whole framework runs anywhere (no TensorFlow / PyTorch dependency). It provides
exactly what the framework needs:

    encode(X)      -> latent codes Z  (the k-dim bottleneck)
    reconstruct(X) -> X_hat
    recon_error(X) -> per-sample MSE  (available as an optional drift signal)

Architecture:  p -> [hidden...] -> k (linear bottleneck) -> [hidden...] -> p
Hidden layers use tanh; the bottleneck and output are linear. Trained with
mini-batch Adam and MSE loss, with early stopping on a held-out tail of the
training data. Inputs are standardised internally using **training statistics
only** (no leakage).

If PyTorch happens to be installed you may swap in a torch model behind the same
three methods; the rest of the framework is agnostic to the backend.
"""

from __future__ import annotations
import numpy as np


class _Dense:
    """A single fully-connected layer with its own Adam optimiser state."""

    def __init__(self, n_in: int, n_out: int, activation: str, rng: np.random.Generator):
        limit = np.sqrt(6.0 / (n_in + n_out))          # Glorot uniform
        self.W = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation
        # Adam moments
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.z = X @ self.W + self.b
        if self.activation == "tanh":
            self.a = np.tanh(self.z)
        elif self.activation == "relu":
            self.a = np.maximum(0.0, self.z)
        else:  # linear
            self.a = self.z
        return self.a

    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            grad_z = grad_a * (1.0 - self.a ** 2)
        elif self.activation == "relu":
            grad_z = grad_a * (self.z > 0)
        else:
            grad_z = grad_a
        m = self.X.shape[0]
        self.gW = (self.X.T @ grad_z) / m
        self.gb = grad_z.mean(axis=0)
        return grad_z @ self.W.T

    def step(self, lr: float, t: int, b1=0.9, b2=0.999, eps=1e-8):
        self.mW = b1 * self.mW + (1 - b1) * self.gW
        self.vW = b2 * self.vW + (1 - b2) * (self.gW ** 2)
        self.W -= lr * (self.mW / (1 - b1 ** t)) / (np.sqrt(self.vW / (1 - b2 ** t)) + eps)
        self.mb = b1 * self.mb + (1 - b1) * self.gb
        self.vb = b2 * self.vb + (1 - b2) * (self.gb ** 2)
        self.b -= lr * (self.mb / (1 - b1 ** t)) / (np.sqrt(self.vb / (1 - b2 ** t)) + eps)


class Autoencoder:
    def __init__(self,
                 n_features: int,
                 latent_dim: int | None = None,
                 hidden: tuple[int, ...] | None = None,
                 lr: float = 1e-2,
                 epochs: int = 300,
                 batch_size: int = 64,
                 val_fraction: float = 0.15,
                 patience: int = 25,
                 seed: int = 42,
                 verbose: bool = False):
        self.p = int(n_features)
        self.k = int(latent_dim) if latent_dim else max(2, self.p // 2)
        if hidden is None:
            h = max(self.k + 1, min(32, 2 * self.p))
            hidden = (h,)
        self.hidden = tuple(int(x) for x in hidden)
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        # Build encoder + decoder layer stacks.
        self.enc_layers: list[_Dense] = []
        dims = [self.p, *self.hidden, self.k]
        for i in range(len(dims) - 1):
            act = "tanh" if i < len(dims) - 2 else "linear"   # linear bottleneck
            self.enc_layers.append(_Dense(dims[i], dims[i + 1], act, self.rng))

        self.dec_layers: list[_Dense] = []
        dims = [self.k, *reversed(self.hidden), self.p]
        for i in range(len(dims) - 1):
            act = "tanh" if i < len(dims) - 2 else "linear"   # linear output
            self.dec_layers.append(_Dense(dims[i], dims[i + 1], act, self.rng))

        self._mu = np.zeros(self.p)
        self._sigma = np.ones(self.p)
        self.history_: list[float] = []

    # ------------- internal standardisation (training stats only) -------------
    def _standardize_fit(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma = np.where(self._sigma < 1e-8, 1.0, self._sigma)

    def _standardize(self, X):
        return (X - self._mu) / self._sigma

    # ------------- forward passes -------------
    def _encode_std(self, Xs):
        a = Xs
        for layer in self.enc_layers:
            a = layer.forward(a)
        return a

    def _decode(self, Z):
        a = Z
        for layer in self.dec_layers:
            a = layer.forward(a)
        return a

    def _forward(self, Xs):
        return self._decode(self._encode_std(Xs))

    # ------------- training -------------
    def fit(self, X_train: np.ndarray) -> "Autoencoder":
        X_train = np.asarray(X_train, dtype=float)
        self._standardize_fit(X_train)
        Xs = self._standardize(X_train)

        n = Xs.shape[0]
        n_val = max(1, int(self.val_fraction * n))
        Xtr, Xval = Xs[:-n_val], Xs[-n_val:]        # causal validation tail
        if Xtr.shape[0] < self.batch_size:
            self.batch_size = max(1, Xtr.shape[0])

        best_val = np.inf
        best_state = None
        bad = 0
        t = 0
        for epoch in range(1, self.epochs + 1):
            perm = self.rng.permutation(Xtr.shape[0])
            for s in range(0, Xtr.shape[0], self.batch_size):
                batch = Xtr[perm[s:s + self.batch_size]]
                out = self._forward(batch)
                # MSE gradient wrt output
                grad = 2.0 * (out - batch) / batch.shape[0]
                for layer in reversed(self.dec_layers):
                    grad = layer.backward(grad)
                for layer in reversed(self.enc_layers):
                    grad = layer.backward(grad)
                t += 1
                for layer in (*self.dec_layers, *self.enc_layers):
                    layer.step(self.lr, t)

            val_pred = self._forward(Xval)
            val_mse = float(np.mean((val_pred - Xval) ** 2))
            self.history_.append(val_mse)
            if self.verbose and (epoch % 25 == 0 or epoch == 1):
                print(f"  [AE] epoch {epoch:4d}  val_mse={val_mse:.5f}")

            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_state = self._snapshot()
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  [AE] early stop at epoch {epoch} (best val_mse={best_val:.5f})")
                    break
        if best_state is not None:
            self._restore(best_state)
        self.best_val_mse_ = best_val
        return self

    # ------------- public API -------------
    def encode(self, X: np.ndarray) -> np.ndarray:
        return self._encode_std(self._standardize(np.asarray(X, dtype=float)))

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(np.asarray(X, dtype=float))
        recon_std = self._forward(Xs)
        return recon_std * self._sigma + self._mu

    def recon_error(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(np.asarray(X, dtype=float))
        recon_std = self._forward(Xs)
        return np.mean((recon_std - Xs) ** 2, axis=1)

    # ------------- snapshot helpers for early stopping -------------
    def _snapshot(self):
        return ([(l.W.copy(), l.b.copy()) for l in self.enc_layers],
                [(l.W.copy(), l.b.copy()) for l in self.dec_layers])

    def _restore(self, state):
        enc, dec = state
        for layer, (W, b) in zip(self.enc_layers, enc):
            layer.W, layer.b = W, b
        for layer, (W, b) in zip(self.dec_layers, dec):
            layer.W, layer.b = W, b
