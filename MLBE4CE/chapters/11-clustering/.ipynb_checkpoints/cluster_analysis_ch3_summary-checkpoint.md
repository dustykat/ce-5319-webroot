
# 📘 Chapter 3 Summary – *Measurement of Proximity*

> _Adapted from: Everitt, B. S., Landau, S., Leese, M., & Stahl, D. (2011). **Cluster Analysis** (5th ed.). John Wiley & Sons._  
> _Link to source: [Wiley Online Library](https://www.wiley.com/en-us/Cluster+Analysis%2C+5th+Edition-p-9780470749913)_

---

## 📌 Overview

Clustering depends heavily on how we define the "closeness" of individuals—called **proximity**. Proximity can be expressed as:
- **Dissimilarity** or **distance** (higher = more different)
- **Similarity** (higher = more alike)

---

## 3.1 Direct vs Indirect Proximities

- **Direct**: Obtained from subjective evaluations (e.g., taste testing).
- **Indirect**: Derived from variable data using mathematical formulas (most common).
- Example: Crime rate data used to compute Euclidean distances between cities.

---

## 3.2 Proximities for Categorical Data

### Binary Variables
- Match types: a (1–1), b (1–0), c (0–1), d (0–0)
- Several coefficients exist:
  - **Matching coefficient** (S1): includes co-absences
  - **Jaccard coefficient** (S2): ignores co-absences
  - **Rogers–Tanimoto**, **Sneath–Sokal**, **Gower–Legendre** variants

⚠️ Choice of similarity measure depends on the data context (e.g., presence/absence, symmetric variables).

### Multi-level Categorical Variables
- One-hot encoding not preferred due to excessive 0–0 matches.
- Instead, use per-variable match scoring (1 for same, 0 for different).

---

## 3.3 Proximities for Continuous Data

### Distance Measures
- **Euclidean (D1)**: Physical geometric distance
- **City Block / Manhattan (D2)**: Sum of absolute differences
- **Minkowski (D3)**: Generalized form of D1 and D2
- **Canberra (D4)**: Sensitive to small values

### Correlation-based Measures
- **Pearson (D5)**: Similarity of profiles (ignores magnitude)
- **Angular Separation (D6)**: Cosine similarity

📎 Note: Correlation-based measures may mislead if scale matters.

---

## 3.4 Mixed-Type Data

- **Gower's coefficient** (1971) is preferred.
- Handles binary, categorical, and continuous data with custom similarity scores.
- Common in survey data and bioinformatics.

---

## 3.5 Structured / Repeated Measures Data

- Includes repeated observations over time, conditions, or spatial locations.
- Strategies:
  - Summarize time series (e.g., slope and intercept from regression)
  - Compute similarities on per-condition means
  - Use **Levenshtein (edit) distance** or **Jaro similarity** for sequence data

---

## 3.6 Group-to-Group Proximities

- Needed in hierarchical clustering.
- Approaches:
  - **Single linkage**: minimum distance (nearest neighbor)
  - **Complete linkage**: maximum distance (farthest neighbor)
  - **Group average**: mean pairwise distance
  - **Centroid / Mahalanobis distance**: takes into account shape and spread

---

## 🔁 Transforming to Euclidean Distance

- Some dissimilarity matrices are not Euclidean.
- Techniques exist (e.g., **Gower's transformation**, **Lingoes**, **Cailliez**) to convert them.

---

## 📦 Software Notes

- Most modern tools (R, Stata, SPSS, SAS) support the proximity measures discussed.
- **R packages**: `cluster`, `proxy`, `clusterSim`

---

## 🧠 Key Takeaways

- Always match the proximity measure to your **data type** and **clustering goal**.
- Misusing a similarity or distance metric can drastically distort your results.
- Be cautious with correlation-based measures and ensure scale compatibility.

