"""Keras model factory for the three predictor architectures (MLP, RNN, CNN).

Deliberately does NOT `import keras` at module level. Determinism-related
environment variables (TF_DETERMINISTIC_OPS, TF_CUDNN_DETERMINISTIC,
TF_ENABLE_ONEDNN_OPTS) must be set before TensorFlow/Keras is first imported
anywhere in the process -- see mabss.seeding.set_global_seed. Importing keras
lazily inside `create_model` (rather than at module scope) is what makes that
ordering possible for callers that set env vars first.
"""

from __future__ import annotations


def create_model(arch: str, random_seed: int, input_dim: int):
    """
    Ported from MABSS_utility.py:62-90 (`create_model`), with one deliberate
    change: ReLU activations added to the MLP's two hidden Dense layers.

    Why: the original MLP branch was three consecutive Dense layers with no
    activation function at all (`activation` defaults to linear in Keras). A
    composition of purely linear maps collapses to a single linear map, so
    every "MLP" arm was, mathematically, an over-parameterized linear
    regression -- not the "Deep Multi-Layer Perceptron" Manuscript.tex:208
    describes. The RNN branch already gets nonlinearity from SimpleRNN's
    default tanh activation, and the CNN branch already uses relu explicitly
    (line 81-82 of the original) -- only the MLP branch was affected.

    This was a deliberate, discussed decision (not a silent "bug fix"): making
    the MLP a real nonlinear network invalidates every cached prediction in
    experiments/ and experiments_cluster/, since the homogeneous "50 DMLP"
    pool and the MLP third of the heterogeneous pool were previously linear.
    Regenerating the affected artifacts (walk_forward_training reruns, then the
    downstream bandit reruns) is a Toeplitz/GPU job -- see the plan's Phase 5/6,
    explicitly out of scope for this change. This function is the corrected
    implementation that job should use; it is not itself a re-run.

    The final output Dense(1) layer is left without an activation in all three
    architectures (a linear regression head predicting a continuous return),
    matching the original's convention and standard practice for a regression
    output layer.
    """
    import keras

    def _init():
        return keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)

    if arch == "mlp":
        model = keras.Sequential(
            [
                keras.Input(shape=(input_dim,)),
                keras.layers.Dense(64, activation="relu", kernel_initializer=_init()),
                keras.layers.Dense(64, activation="relu", kernel_initializer=_init()),
                keras.layers.Dense(1, kernel_initializer=_init()),
            ]
        )
    elif arch == "rnn":
        model = keras.Sequential(
            [
                keras.Input(shape=(input_dim, 1)),
                keras.layers.SimpleRNN(64, return_sequences=True, kernel_initializer=_init()),
                keras.layers.SimpleRNN(64, kernel_initializer=_init()),
                keras.layers.Dense(1, kernel_initializer=_init()),
            ]
        )
    elif arch == "cnn":
        model = keras.Sequential(
            [
                keras.Input(shape=(input_dim, 1)),
                keras.layers.Conv1D(64, 3, activation="relu", kernel_initializer=_init()),
                keras.layers.Conv1D(64, 3, activation="relu", kernel_initializer=_init()),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(1, kernel_initializer=_init()),
            ]
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model
