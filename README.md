# ML Classification Project — Heart Disease & Rain Prediction

Two-part machine learning project applying Decision Trees, MLPs, PCA, K-Means clustering, and ensemble model comparison to real-world classification tasks. Built with scikit-learn on UCI Heart Disease data (Part 3a) and Australian weather data (Part 3b).
Created by ECE M148 at UCLA. SPRING 2025

`SEED = 66` is used throughout for reproducibility.

---

## Part 3a — Heart Disease Classification

### Dataset

UCI Heart Disease dataset (`datasets/heartdisease.csv`), 14 attributes, binary target (`sick` → `target`: 0 = healthy, 1 = sick).

| Split | Healthy | Sick | Total |
|-------|---------|------|-------|
| Train | 107 | 89 | 196 |
| Test (35%) | 58 | 49 | 107 |

Majority-class baseline accuracy: **0.542**

### Preprocessing

- `LabelEncoder` on `sex` and target `sick`
- **Numerical features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): MinMaxScaler
- **Categorical features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`): OneHotEncoder
- Train-test split: 35% test, `stratify=y`, `random_state=0`
- Output: **30 features** after transformation

---

### 1. Decision Trees

#### 1.1 Default Decision Tree (train set)

| Metric | Value |
|--------|-------|
| Train accuracy | 1.000 (perfect — overfits) |
| Confusion matrix | [[107, 0], [0, 89]] |

#### 1.2 Tree Visualization (first 2 layers)

| Node | Split | Gini | Samples |
|------|-------|------|---------|
| Root | `cat__cp_0 ≤ 0.5` | 0.496 | 100% |
| Left child | `num__oldpeak ≤ 0.427` | 0.305 | 49% |
| Right child | `cat__ca_0 ≤ 0.5` | 0.412 | 51% |

**Gini improvement of root split:**

```
0.496 − ((0.49 × 0.305) + (0.51 × 0.412)) = 0.137
```

The first split on chest pain type (`cp`) yields a substantial impurity reduction of 0.137, confirming it is the most discriminative feature.

**Feature importances:** 13 of 30 features have non-zero importance. Removing zero-importance features does not change the decision tree.

#### 1.3 Optimized Decision Tree (GridSearchCV, 5-fold CV)

| Hyperparameter | Values searched |
|----------------|----------------|
| `max_depth` | [4, 8, 16, 32, 64] |
| `min_samples_split` | [4, 8, 16, 32] |
| `criterion` | gini, entropy, log_loss |

**Best parameters:** `criterion=entropy`, `max_depth=4`, `min_samples_split=16`  
**Best CV score:** 0.7547

| Metric | Value |
|--------|-------|
| Test accuracy | **0.7851** |
| Confusion matrix | [[50, 8], [15, 34]] |

---

### 2. Multi-Layer Perceptron

**Parameters:** `hidden_layer_sizes=(100, 100)`, `max_iter=1000`, `random_state=SEED`

| Split | Accuracy | Confusion Matrix |
|-------|----------|-----------------|
| Train | 1.000 | [[107, 0], [0, 89]] |
| Test | **0.8318** | [[52, 6], [12, 37]] |

#### Speed Comparison (default DT vs. MLP)

| Model | Train time | Predict time |
|-------|-----------|--------------|
| Decision Tree | 0.008s | 0.00072s |
| MLP | 2.272s | 0.00087s |

Decision Tree is faster for both training and prediction. MLP advantage: captures non-linear relationships. MLP disadvantage: lacks interpretability.

---

### 3. PCA

**Configuration:** `n_components=8`, `random_state=SEED`

#### Explained Variance by Component

| PC | Variance Explained (%) |
|----|------------------------|
| 1 | 23.40 |
| 2 | 13.97 |
| 3 | 10.09 |
| 4 | 8.11 |
| 5 | 7.51 |
| 6 | 6.75 |
| 7 | 5.99 |
| 8 | 4.95 |

#### PCA + Decision Tree

| Metric | Value |
|--------|-------|
| Test accuracy | **0.7851** |
| Confusion matrix | [[50, 8], [15, 34]] |

Same accuracy as without PCA — the top 8 components preserve all discriminative information. PCA successfully removed redundant features.

#### PCA + MLP

| Metric | Value |
|--------|-------|
| Test accuracy | **0.8131** |
| Confusion matrix | [[52, 6], [14, 35]] |

~2% lower than MLP without PCA (0.8318). MLP captures non-linear relationships that PCA's linear projections partially discard.

---

### 4. K-Means Clustering

**K=15 on train data — Inertia: 385.51**

#### Elbow Method Results

| Data | Optimal cluster range |
|------|-----------------------|
| Raw train | k = 11 to 14 |
| PCA-reduced train | k = 7 to 11 |

PCA-reduced data produces lower inertia values overall and a smoother elbow curve that flattens earlier, indicating good clustering can be achieved with fewer clusters after dimensionality reduction.

---

## Part 3b — Rain Prediction

### Dataset

`datasets/weather_data.csv` — 23 columns of Australian weather observations. Target: `RainTomorrow` (Yes / No).

### Preprocessing

| Step | Strategy |
|------|----------|
| Dropped columns | `Date`, `RainTomorrow` (used as target) |
| Missing targets | Rows dropped |
| Missing numerical | Filled with **median** (robust to outliers) |
| Missing categorical | Filled with **mode** |
| Feature augmentation | `TempRange = MaxTemp − MinTemp`; `PressureChange = Pressure3pm − Pressure9am`; `HumidityChange = Humidity3pm − Humidity9am` |
| Scaling | `StandardScaler` on all numerical features |
| Encoding | `OneHotEncoder(drop='first')` on categorical (avoids multicollinearity) |
| Train-test split | 80/20, stratified |
| Class balancing | Undersample majority class (`No`) to match minority class (`Yes`) |

### Models & Results

All models tuned with `GridSearchCV`. Evaluation uses both accuracy and F1 score (`pos_label="Yes"`).

#### Logistic Regression

| Hyperparameter | Values searched |
|----------------|----------------|
| `C` | [0.1, 1.0, 10.0] |
| `penalty` | l1, l2 |
| `solver` | liblinear, saga |

**Best:** `C=1.0`, `penalty=l1`, `solver=liblinear`

| Metric | Value |
|--------|-------|
| Test accuracy | 0.7927 |
| Test F1 | **0.6267** |

#### Decision Tree

| Hyperparameter | Values searched |
|----------------|----------------|
| `max_depth` | [5, 10, 20, None] |
| `min_samples_split` | [10, 20, 30, 50] |
| `min_samples_leaf` | [2, 4, 8, 16] |
| `criterion` | gini, entropy |

**Best:** `criterion=gini`, `max_depth=10`, `min_samples_leaf=16`, `min_samples_split=50`

| Metric | Value |
|--------|-------|
| Test accuracy | 0.7793 |
| Test F1 | **0.6032** |

#### K-Nearest Neighbors

| Hyperparameter | Values searched |
|----------------|----------------|
| `n_neighbors` | [3, 5, 7, 9] |
| `weights` | uniform, distance |
| `metric` | euclidean, manhattan |

**Best:** `metric=euclidean`, `n_neighbors=9`, `weights=distance`

| Metric | Value |
|--------|-------|
| Test accuracy | 0.7955 |
| Test F1 | **0.6249** |

#### Model Comparison Summary

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.7927 | 0.6267 |
| KNN | 0.7955 | 0.6249 |
| Decision Tree | 0.7793 | 0.6032 |

Logistic Regression achieves the best F1 score. All three models exceed the 60% F1 threshold required for extra credit.

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
```

```python
# datasets/
#   heartdisease.csv   (Part 3a)
#   weather_data.csv   (Part 3b)
# helper.py            (draw_confusion_matrix, heatmap, make_meshgrid, plot_contours, draw_contour)
```
