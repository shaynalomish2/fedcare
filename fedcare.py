import numpy as np
import matplotlib

matplotlib.use("Agg")  # lets the script save plots even without a screen
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------------
# SETTINGS  (change these to experiment!)
# ----------------------------------------------------------------------------
SEED = 42            # makes results repeatable
NUM_HOSPITALS = 5    # how many "hospitals" take part
ROUNDS = 30          # how many federated rounds (send / aggregate / send back)
LOCAL_EPOCHS = 5     # how long each hospital trains per round
LEARNING_RATE = 0.1  # step size of the learning
NON_IID = True       # True = hospitals have DIFFERENT patient mixes (realistic)
DIRICHLET_ALPHA = 0.5  # smaller = more different hospitals (0.1 very different, 10 almost equal)
CSV_PATH = None      # e.g. "heart.csv" to use your own dataset (see load_csv below)
CSV_TARGET = "target"  # name of the 0/1 label column in your CSV


# ----------------------------------------------------------------------------
# STEP 1: LOAD DATA
# ----------------------------------------------------------------------------
def load_builtin():
    """Breast Cancer Wisconsin dataset (569 patients, 30 measurements each).
    sklearn labels malignant=0, benign=1. We flip it so 1 = DISEASE,
    which makes 'recall' mean 'how many sick patients did we catch?'"""
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target
    return X, y


def load_csv(path, target):
    """Use your own dataset, e.g. Heart Disease from UCI/Kaggle.
    Needs numeric columns plus one label column called `target`.
    Rows with missing values are dropped. Any label > 0 counts as 'disease' (1),
    so the UCI Heart Disease label (0-4) automatically becomes 0/1."""
    import pandas as pd

    df = pd.read_csv(path, na_values=["?"]).dropna()
    y = (df[target].astype(float) > 0).astype(int).values
    X = df.drop(columns=[target]).select_dtypes("number").values
    return X, y


def prepare_data():
    X, y = load_csv(CSV_PATH, CSV_TARGET) if CSV_PATH else load_builtin()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    # Scaling puts all measurements on a similar range (helps learning).
    # Simplification: we fit the scaler on all training data. In real life each
    # hospital would scale locally or share only mean/std.
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


# ----------------------------------------------------------------------------
# STEP 2: SPLIT INTO "HOSPITALS"
# ----------------------------------------------------------------------------
def split_into_hospitals(X, y, n, non_iid, alpha, rng):
    """Give each hospital its own slice of the data.
    non_iid=True -> hospitals get different proportions of sick/healthy patients,
    like real life (e.g. a cancer center vs. a general clinic)."""
    if not non_iid:
        chunks = np.array_split(rng.permutation(len(y)), n)
        return [(X[c], y[c]) for c in chunks]

    while True:  # retry until every hospital has a reasonable number of patients
        idx_per_hospital = [[] for _ in range(n)]
        for cls in np.unique(y):
            cls_idx = rng.permutation(np.where(y == cls)[0])
            proportions = rng.dirichlet(alpha * np.ones(n))
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            for h, part in enumerate(np.split(cls_idx, cuts)):
                idx_per_hospital[h].extend(part.tolist())
        if min(len(i) for i in idx_per_hospital) >= 20:
            break
    return [(X[np.array(i)], y[np.array(i)]) for i in idx_per_hospital]


# ----------------------------------------------------------------------------
# STEP 3: THE MODEL (logistic regression written from scratch)
# ----------------------------------------------------------------------------
# The "model" is just a list of numbers: one WEIGHT per measurement + one BIAS.
# We store them in a single vector `params`. This vector is what gets shared
# in federated learning. It contains no patient records.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(params, X):
    return sigmoid(X @ params[:-1] + params[-1])


def local_train(params, X, y, epochs, lr, rng, batch_size=32):
    """Train on ONE hospital's private data. Returns the updated weights.
    (The data X, y never leaves this function / this hospital.)"""
    p = params.copy()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch_size):
            b = order[s : s + batch_size]
            error = predict_proba(p, X[b]) - y[b]       # how wrong are we?
            p[:-1] -= lr * (X[b].T @ error) / len(b)    # nudge weights
            p[-1] -= lr * error.mean()                   # nudge bias
    return p


def evaluate(params, X_test, y_test):
    pred = (predict_proba(params, X_test) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


# ----------------------------------------------------------------------------
# STEP 4: AGGREGATION (the "federated" part, done by the cloud server)
# ----------------------------------------------------------------------------
def aggregate(updates, sizes, priorities=None):
    """Combine the hospitals' weight vectors into ONE global model.

    Standard FedAvg : hospitals with more patients count more.
    Priority-aware  : (OUR NOVELTY) hospitals that serve higher-risk patients
                      get an extra multiplier, so the global model pays more
                      attention to the sick-patient pattern (better recall).
    """
    w = np.array(sizes, dtype=float)
    if priorities is not None:
        w = w * np.array(priorities, dtype=float)
    w = w / w.sum()
    return sum(wi * u for wi, u in zip(w, updates))


def run_federated(hospitals, X_test, y_test, rounds, use_priority, rng, verbose=False):
    d = hospitals[0][0].shape[1]
    global_params = np.zeros(d + 1)
    sizes = [len(y) for _, y in hospitals]

    # Priority = 1 + fraction of sick patients at that hospital.
    # (Only ONE small number per hospital is shared, not patient data.)
    priorities = [1.0 + y.mean() for _, y in hospitals] if use_priority else None

    history = []
    for r in range(rounds):
        updates = []
        for X_k, y_k in hospitals:                      # each hospital, privately...
            updates.append(local_train(global_params, X_k, y_k, LOCAL_EPOCHS, LEARNING_RATE, rng))
        if verbose and r == 0:
            print("   What hospital 1 actually SENDS to the cloud (first 5 numbers):")
            print("  ", np.round(updates[0][:5], 4), "...  <- just weights, no patient data!")
        global_params = aggregate(updates, sizes, priorities)   # ...cloud combines
        history.append(evaluate(global_params, X_test, y_test))
    return global_params, history


# ----------------------------------------------------------------------------
# STEP 5: RUN EVERYTHING AND COMPARE
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    X_train, X_test, y_train, y_test = prepare_data()
    hospitals = split_into_hospitals(X_train, y_train, NUM_HOSPITALS, NON_IID, DIRICHLET_ALPHA, rng)
    d = X_train.shape[1]

    print("=" * 66)
    print(" FedCare - Federated Learning for Privacy-Preserving Disease Prediction")
    print("=" * 66)
    print(f"Total training patients: {len(y_train)} | test patients: {len(y_test)} | features: {d}")
    print(f"Data type: {'NON-IID (different hospitals)' if NON_IID else 'IID (similar hospitals)'}\n")
    print("Hospital   Patients   Sick patients (%)")
    for i, (_, y_k) in enumerate(hospitals, 1):
        print(f"   H{i}        {len(y_k):4d}        {100 * y_k.mean():5.1f}%")
    print()

    total_epochs = ROUNDS * LOCAL_EPOCHS

    # A. Centralized baseline (privacy NOT protected)
    central = local_train(np.zeros(d + 1), X_train, y_train, total_epochs, LEARNING_RATE, rng)
    res_central = evaluate(central, X_test, y_test)

    # B. Local-only baseline (each hospital alone)
    local_results = []
    for X_k, y_k in hospitals:
        p = local_train(np.zeros(d + 1), X_k, y_k, total_epochs, LEARNING_RATE, rng)
        local_results.append(evaluate(p, X_test, y_test))
    res_local = {k: float(np.mean([r[k] for r in local_results])) for k in res_central}

    # C. Federated learning
    print("--- Federated training (privacy preserved) ---")
    _, hist_fedavg = run_federated(hospitals, X_test, y_test, ROUNDS, False, rng, verbose=True)
    _, hist_prio = run_federated(hospitals, X_test, y_test, ROUNDS, True, rng)
    res_fedavg, res_prio = hist_fedavg[-1], hist_prio[-1]

    # ---- results table ----
    print("\n" + "=" * 66)
    print(f"{'Method':<34}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>8}")
    print("-" * 66)
    for name, r in [
        ("A. Centralized (no privacy)", res_central),
        ("B. Local-only (avg of hospitals)", res_local),
        ("C1. Federated - FedAvg", res_fedavg),
        ("C2. Federated - Priority-aware", res_prio),
    ]:
        print(f"{name:<34}{r['accuracy']:>9.3f}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")
    print("=" * 66)
    print("Recall = of all truly sick patients, how many did the model catch?")

    # ---- sustainability: how much data travelled? ----
    raw_bytes = X_train.nbytes + y_train.nbytes
    fl_bytes = ROUNDS * NUM_HOSPITALS * 2 * (d + 1) * 8  # up + down, 8 bytes per number
    print("\n--- Network data moved (sustainability) ---")
    print(f"Sending all raw patient data once : {raw_bytes / 1024:8.1f} KB  (and exposes patients!)")
    print(f"Federated learning, {ROUNDS} rounds       : {fl_bytes / 1024:8.1f} KB  (weights only)")
    print("Note: with a tiny dataset like this, raw data is small. With millions of")
    print("images/ECG streams, raw data is huge while weights stay tiny - that is the saving.")

    # ---- plot ----
    rounds_axis = np.arange(1, ROUNDS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, metric, title in zip(ax, ["accuracy", "recall"], ["Accuracy", "Recall (sick patients caught)"]):
        a.plot(rounds_axis, [h[metric] for h in hist_fedavg], label="Federated - FedAvg", lw=2)
        a.plot(rounds_axis, [h[metric] for h in hist_prio], label="Federated - Priority-aware (ours)", lw=2)
        a.axhline(res_central[metric], color="green", ls="--", label="Centralized (no privacy)")
        a.axhline(res_local[metric], color="red", ls=":", label="Local-only (avg)")
        a.set_xlabel("Federated round")
        a.set_ylabel(title)
        a.set_title(f"{title} vs. training rounds")
        a.grid(alpha=0.3)
    ax[0].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("fedcare_results.png", dpi=150)
    print("\nSaved graph -> fedcare_results.png")


if __name__ == "__main__":
    main()