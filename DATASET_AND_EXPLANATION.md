# FedCare: Dataset and Detailed Explanation

---

# PART 1: THE DATASET

## 1.1 Which dataset and where it comes from

| | |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic), often called "WDBC" |
| **Original home** | UCI Machine Learning Repository (University of California, Irvine) |
| **Page** | https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic |
| **Creators** | William Wolberg, Olvi Mangasarian, Nick Street, W. Street |
| **Donated** | 31 Oct 1995 (scikit-learn's description lists donor Nick Street, November 1995) |
| **Size** | 569 patients, 30 numeric features, no missing values |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0): free to use, you must credit the authors |
| **DOI** | 10.24432/C5DW2B |
| **Associated paper** | Street, W., Wolberg, W., Mangasarian, O. (1993), "Nuclear feature extraction for breast tumor diagnosis", Electronic Imaging |

**Cite it in your report like this:**
> Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## 1.2 How I got it (so you know exactly what you are using)

1. The data ships **inside the scikit-learn library**: `sklearn.datasets.load_breast_cancer()`. scikit-learn's own documentation says it is "a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets".
2. I read the UCI page and confirmed the authors, size (569 patients, 30 features), no missing values, license and DOI.
3. I counted the classes in the file: **212 malignant, 357 benign**. (The UCI page did not list this; scikit-learn's description and my count agree.)
4. I exported that copy to the CSV file `data/breast_cancer_wisconsin_diagnostic.csv`, and checked that loading the CSV into the project gives exactly the same results as the built-in loader.

**Honest limits of this check:** I did not download the original file from UCI itself. I verified the copy against UCI's description of it (size, features, authors), not row by row. If a reviewer asks, you can download the original from the UCI page above and compare, or simply say "the dataset is the UCI WDBC dataset, accessed via scikit-learn".

## 1.3 What is in each row

Each row is **one patient's breast lump**. Doctors took a tiny sample of the lump with a thin needle (a "fine needle aspirate"), photographed the cells under a microscope, and software measured the cell nuclei in the image. The 30 numbers are those measurements. The label says whether the lump turned out to be cancerous.

**10 measurements, each recorded 3 ways = 30 columns:**

| Measurement | What it describes (plain words) |
|---|---|
| radius | Average distance from the centre of the nucleus to its edge (size) |
| texture | How much the grey shades vary inside the nucleus |
| perimeter | Length of the nucleus outline |
| area | Area of the nucleus |
| smoothness | How smooth or jagged the outline is (local variation in radius) |
| compactness | How "squeezed" the shape is (perimeter squared / area) |
| concavity | How deep the dents in the outline are |
| concave points | How many dents there are |
| symmetry | How symmetrical the nucleus is |
| fractal dimension | How rough the outline is ("coastline" roughness) |

The 3 ways (column prefixes in the CSV):
- `mean_...` : average over all nuclei in the image
- `...error` : standard error (how much that measurement varies)
- `worst_...` : the average of the three largest values (the "most extreme" nuclei)

Example column names: `mean_radius`, `radius_error`, `worst_radius`.

**The label column:** `diagnosis_malignant`
- `1` = malignant (cancer) = "sick"  → 212 patients
- `0` = benign (not cancer) = "healthy" → 357 patients

(scikit-learn stores it the other way round, 0 = malignant. I flipped it so that **1 = disease**, which makes "recall" mean "how many sick patients did we catch".)

**First row of the CSV, as a sanity check:** `mean_radius = 17.99, mean_texture = 10.38, mean_perimeter = 122.8, mean_area = 1001.0 ...`

## 1.4 Is this dataset a good choice for your project?

**Good:**
- Real medical data, small, clean, no missing values, runs in seconds on a laptop.
- Public and license allows student use with credit.
- The most common beginner benchmark, so reviewers know it.
- Labels are reliable (confirmed diagnoses).

**Be upfront about:**
- It covers **one disease** (breast cancer), not "disease prediction in general". Say: "proof of concept on breast cancer; the same pipeline applies to other diseases."
- It is **one dataset cut into 5 pretend hospitals**. Real hospitals would have different equipment, formats and data quality.
- 569 patients is small; a 114-patient test set means one patient is almost 1% accuracy.
- The data is from the 1990s and one research group. Fine for a method demonstration, not for clinical claims.

## 1.5 A second dataset (optional): Heart Disease

| | |
|---|---|
| **Name** | Heart Disease (UCI) |
| **Page** | https://archive.ics.uci.edu/dataset/45/heart+disease |
| **Creators** | Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano |
| **Size** | 303 patients, 13 features, **has missing values** |
| **License / DOI** | CC BY 4.0 / 10.24432/C52P4X |
| **Label** | `num`: 0 = no disease, 1-4 = disease present |
| **Real multi-hospital origin** | Four sources: Cleveland, Hungary, Switzerland, VA Long Beach |

Why it is interesting: it genuinely comes from **different hospitals**, which fits a federated story well. The page notes that the Cleveland file is the one most researchers use.

How to use it: download the processed Cleveland file from the UCI page, save it as a CSV with a header row, and set in `fedcare.py`:
```python
CSV_PATH = "heart.csv"
CSV_TARGET = "num"      # or whatever you named the label column
```
The loader now drops rows with missing values (`?` is treated as missing) and turns any label above 0 into "disease", so the 0-4 label becomes 0/1 automatically. I tested the loader on the breast cancer CSV, not on a downloaded heart file, so check that your column names match.

(Pima Indians Diabetes is another common choice, usually downloaded from Kaggle. I did not verify its page here, so check its license and description before citing it.)

---

# PART 2: DETAILED EXPLANATION OF THE PROJECT

## 2.1 The problem in one story

Imagine 5 hospitals that each want an AI that spots cancer early. Each hospital has only a few hundred patients, so its AI is mediocre. If they pooled all their data the AI would be much better, but **laws and ethics (HIPAA, GDPR, India's DPDP Act) forbid sending patient records to a shared server**. Stuck.

**Federated learning** solves this by moving the *learning* to the data instead of moving the data to the learning.

## 2.2 What "learning" physically is

Our model is **logistic regression**, the simplest classifier. For one patient with measurements `x1 ... x30`:

1. Score: `z = w1·x1 + w2·x2 + ... + w30·x30 + b`
2. Probability of cancer: `p = 1 / (1 + e^(-z))` (always between 0 and 1)
3. If `p ≥ 0.5` predict "malignant", else "benign".

The model is just the **31 numbers** `w1...w30, b`. Learning = adjusting those numbers so predictions match the true diagnoses.

**One training step (gradient descent), in words:** predict for a small batch of patients, measure how wrong you were (`error = p - truth`), then nudge each weight a little in the direction that reduces the error. Repeat thousands of times. This is `local_train()` in the code.

**Why this is private:** after training, the 31 numbers say things like "bigger, more irregular nuclei push the score towards cancer". They do not contain any patient's record.

## 2.3 One federated round, step by step

Round r:
1. **Cloud → hospitals:** the cloud sends the current global model (31 numbers) to all 5 hospitals.
2. **Local training:** each hospital trains that model for 5 epochs on its **own** patients. Data never moves.
3. **Hospitals → cloud:** each hospital sends back its updated 31 numbers (and its patient count).
4. **Aggregation:** the cloud averages the 5 sets of numbers into a new global model.
5. Repeat for 30 rounds.

After round 30, the global model has "seen" all hospitals' patterns indirectly, though no patient record ever left a hospital.

## 2.4 Aggregation with a tiny worked example

Say the model has only 2 numbers. Hospital A (100 patients) returns `[0.2, 0.4]`. Hospital B (300 patients) returns `[0.6, 0.0]`.

**Standard FedAvg** weights by patient count (A: 100, B: 300):
- first number: (100×0.2 + 300×0.6) / 400 = **0.50**
- second number: (100×0.4 + 300×0.0) / 400 = **0.10**

**Priority-aware (your novelty).** Multiply each hospital's weight by `1 + fraction of sick patients`. Say A is a cancer centre with 90% sick patients (multiplier 1.9) and B is a general clinic with 10% sick patients (multiplier 1.1):
- effective weights: A = 100×1.9 = 190, B = 300×1.1 = 330, total 520
- first number: (190×0.2 + 330×0.6) / 520 = **0.454**
- second number: (190×0.4 + 330×0.0) / 520 = **0.146**

Result: the cancer centre's opinion counts for more than its size alone would suggest, so the global model pays more attention to what sick patients look like. That is why recall (catching sick patients) can improve.

## 2.5 Why non-IID data matters

"IID" means every hospital sees a similar mix of patients. Real hospitals do not. In the test run the 5 hospitals looked like this:

| Hospital | Patients | Sick patients |
|---|---|---|
| H1 | 30 | 63.3% |
| H2 | 33 | 54.5% |
| H3 | 95 | 100.0% |
| H4 | 263 | 3.4% |
| H5 | 34 | 85.3% |

H3 has only sick patients; H4 has almost none. A hospital that never sees a healthy patient cannot learn what healthy looks like. This is a hard setting for federated learning, and it is exactly why the local-only models were worse (accuracy 0.919) than the federated model (0.974).

## 2.6 What the metrics mean

Out of the 114 test patients, each prediction falls in one of four boxes:

| | Model says sick | Model says healthy |
|---|---|---|
| **Truly sick** | True Positive (caught) | False Negative (**missed cancer**) |
| **Truly healthy** | False Positive (false alarm) | True Negative |

- **Accuracy** = correct predictions / all predictions.
- **Precision** = of those we called sick, how many really were? (avoids false alarms)
- **Recall** = of the truly sick, how many did we catch? (avoids missed cancers)
- **F1** = a single number balancing precision and recall.

In medicine, a missed cancer is usually worse than a false alarm, so **recall is the headline metric**.

## 2.7 Reading your results

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| A. Centralized (no privacy) | 0.974 | 0.976 | 0.952 | 0.964 |
| B. Local-only (avg of hospitals) | 0.919 | 0.889 | 0.929 | 0.900 |
| C1. Federated, FedAvg | 0.974 | 1.000 | 0.929 | 0.963 |
| C2. Federated, priority-aware | 0.982 | 1.000 | 0.952 | 0.976 |

Story to tell:
1. **B is worst:** isolation hurts, even with private data.
2. **C1 equals A on accuracy:** federated learning reaches centralized quality without sharing data.
3. **C2 is slightly better on recall:** the novelty helps a little.

**Be honest about C2.** Over 20 random seeds, priority-aware recall averaged 0.956 versus 0.945 for FedAvg. It was never worse and was better in 7 of 20 runs, but the gain is about one percentage point on a small test set. Say "slight, consistent improvement", not "big improvement".

## 2.8 Questions reviewers may ask (with honest answers)

**Q: Is the data really private if weights are shared?**
Mostly, not perfectly. Research shows that weights can leak small amounts of information (for example through "model inversion" attacks). Real systems add secure aggregation or differential privacy. Your project shows the basic principle; mention this as future work.

**Q: Where is the "fog" and "edge" in this project?**
Each simulated hospital's local training server is a fog node (close to the data). The cloud only aggregates. Your architecture diagram shows this; the code simulates it on one laptop.

**Q: Why logistic regression and not deep learning?**
It is simple, fast, easy to explain, and works well on 30 numeric features. The federated mechanism is identical for neural networks; only the model changes.

**Q: Is this simulation realistic?**
Partly. The data is real but the hospitals are artificial slices of one dataset. Non-IID splitting mimics real differences between hospitals.

**Q: How do you know priority-aware is not just luck?**
I tested 20 random seeds: never worse, better in 7. That is modest evidence, not proof. A bigger dataset would give firmer conclusions.

**Q: What is the sustainability angle?**
Weights are a fixed size (31 numbers here) however many patients there are, while raw data grows with every patient and image. On this tiny dataset the saving is not visible (110 KB raw vs 73 KB federated over 30 rounds), but it becomes large for images and continuous sensor streams. Sending less data also means less network energy. Also, keeping care local helps rural hospitals (SDG 3, 9, 10).

## 2.9 Limitations (put these in your report)

- Single disease, single small dataset, artificial hospital split.
- Simple model; accuracy on complex medical images would need deeper models.
- No formal privacy guarantee (no differential privacy or secure aggregation yet).
- Priority score shares one extra number per hospital (its sick-patient fraction), a small privacy cost.
- Small test set, so differences of 1-2% are within noise.

## 2.10 Where to go next

1. Run the experiments in `GUIDE.md` (IID vs non-IID, more hospitals, more rounds).
2. Try the Heart Disease dataset (section 1.5).
3. Add differential privacy: add small random noise to each hospital's weights before sending and measure the accuracy drop.
4. Move to the Flower framework (`pip install flwr`) once you understand this code.

---

## Sources
- UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- UCI Machine Learning Repository, Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
- scikit-learn dataset description (`load_breast_cancer().DESCR`), read from the installed library
