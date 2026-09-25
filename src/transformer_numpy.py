
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

POLLUTANTS = ["PM10","PM2.5","SO2","CO","O3","NO2"]
FEATURES = [
    "Temperature","Dew Point","Humidity","Wind Speed","Pressure",
    "x_km","y_km","hour","dayofweek",
    "PM10_lag_1","PM2.5_lag_1","SO2_lag_1","CO_lag_1","O3_lag_1","NO2_lag_1",
    "Temperature_lag_1","Dew Point_lag_1","Humidity_lag_1",
    "Wind Speed_lag_1","Pressure_lag_1","Speed_lag_1","Congestion Level_lag_1"
]

class TransformerNumpyEngine:
    """
    Inference-only implementation of the uploaded Keras Transformer weights.

    The notebook trained with tensors reshaped to one timestep (N,1,22), while its
    Input declaration says (24,22). Weight shapes do not depend on timestep length.
    This operational prototype reproduces the one-timestep pathway actually prepared
    in the notebook and therefore avoids pretending a 24-hour sequence was trained.
    """
    def __init__(self, weight_path):
        self.weight_path = Path(weight_path)
        self._load()

    def _load(self):
        with h5py.File(self.weight_path, "r") as f:
            get = lambda path: np.array(f[path], dtype=np.float64)
            # MHA
            self.Wq = get("layers/transformer_block/att/query_dense/vars/0")
            self.bq = get("layers/transformer_block/att/query_dense/vars/1")
            self.Wk = get("layers/transformer_block/att/key_dense/vars/0")
            self.bk = get("layers/transformer_block/att/key_dense/vars/1")
            self.Wv = get("layers/transformer_block/att/value_dense/vars/0")
            self.bv = get("layers/transformer_block/att/value_dense/vars/1")
            self.Wo = get("layers/transformer_block/att/output_dense/vars/0")
            self.bo = get("layers/transformer_block/att/output_dense/vars/1")
            # FFN
            self.W1 = get("layers/transformer_block/ffn/layers/dense/vars/0")
            self.b1 = get("layers/transformer_block/ffn/layers/dense/vars/1")
            self.W2 = get("layers/transformer_block/ffn/layers/dense_1/vars/0")
            self.b2 = get("layers/transformer_block/ffn/layers/dense_1/vars/1")
            # Layer norm
            self.g1 = get("layers/transformer_block/layernorm1/vars/0")
            self.be1 = get("layers/transformer_block/layernorm1/vars/1")
            self.g2 = get("layers/transformer_block/layernorm2/vars/0")
            self.be2 = get("layers/transformer_block/layernorm2/vars/1")
            # Head
            self.Wd = get("layers/dense/vars/0")
            self.bd = get("layers/dense/vars/1")
            self.Wout = get("layers/dense_1/vars/0")
            self.bout = get("layers/dense_1/vars/1")

    @staticmethod
    def _softmax(a, axis=-1):
        a = a - np.max(a, axis=axis, keepdims=True)
        e = np.exp(a)
        return e / np.sum(e, axis=axis, keepdims=True)

    @staticmethod
    def _ln(z, gamma, beta, eps=1e-6):
        m = np.mean(z, axis=-1, keepdims=True)
        v = np.mean((z-m)**2, axis=-1, keepdims=True)
        return (z-m) / np.sqrt(v+eps) * gamma + beta

    def predict_matrix(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.shape[-1] != 22:
            raise ValueError(f"Expected 22 features; got {X.shape[-1]}")

        # Keras MHA projections: [B,T,F] x [F,H,D] -> [B,T,H,D]
        Q = np.einsum("btf,fhd->bthd", X, self.Wq) + self.bq
        K = np.einsum("btf,fhd->bthd", X, self.Wk) + self.bk
        V = np.einsum("btf,fhd->bthd", X, self.Wv) + self.bv

        # Attention scores [B,H,Tq,Tk]
        scores = np.einsum("bthd,bshd->bhts", Q, K) / np.sqrt(Q.shape[-1])
        attn = self._softmax(scores, axis=-1)
        context = np.einsum("bhts,bshd->bthd", attn, V)
        att_out = np.einsum("bthd,hdo->bto", context, self.Wo) + self.bo

        out1 = self._ln(X + att_out, self.g1, self.be1)
        ff = np.maximum(0.0, np.einsum("btf,fd->btd", out1, self.W1) + self.b1)
        ff = np.einsum("btd,df->btf", ff, self.W2) + self.b2
        out2 = self._ln(out1 + ff, self.g2, self.be2)

        last = out2[:, -1, :]
        h = np.maximum(0.0, last @ self.Wd + self.bd)
        return h @ self.Wout + self.bout

    def predict_frame(self, Xdf):
        Xdf = Xdf[FEATURES].astype(float)
        y = self.predict_matrix(Xdf.values)
        return pd.DataFrame(y, columns=POLLUTANTS, index=Xdf.index)
