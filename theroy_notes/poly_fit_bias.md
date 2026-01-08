# Understanding `include_bias` in sklearn PolynomialFeatures

## Overview

The `include_bias` parameter in `PolynomialFeatures` controls whether to add a constant feature (column of 1s) to the transformed feature matrix.

## What Does `include_bias` Do?

### With `include_bias=True` (default)

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=True)
y = [[5]]
transformed = poly.transform(y)
# Result: [[1, 5, 25]]
#          ↑  ↑  ↑
#        bias y  y²
```

The linear model then fits:
```
x = coef[0]·(1) + coef[1]·(y) + coef[2]·(y²) + intercept
```

**Two constant terms:**
- `coef[0]` — coefficient for the bias feature (1)
- `intercept_` — the model's intercept parameter

**Actual constant = `coef[0] + intercept_`**

### With `include_bias=False`

```python
poly = PolynomialFeatures(degree=2, include_bias=False)
y = [[5]]
transformed = poly.transform(y)
# Result: [[5, 25]]
#          ↑  ↑
#          y  y²
```

The linear model fits:
```
x = coef[0]·(y) + coef[1]·(y²) + intercept
```

**One constant term:**
- `intercept_` — the model's intercept parameter (this is c₀)

## When to Use Each Option

### Use `include_bias=True` when:

1. **Using `fit_intercept=False`** in the regressor
   ```python
   from sklearn.linear_model import LinearRegression
   
   model = Pipeline([
       ('poly', PolynomialFeatures(2, include_bias=True)),
       ('linear', LinearRegression(fit_intercept=False))
   ])
   # The bias feature becomes the constant term
   ```

2. **Regularized models** (Ridge, Lasso) where you don't want to penalize the intercept
   ```python
   from sklearn.linear_model import Ridge
   
   model = Pipeline([
       ('poly', PolynomialFeatures(2, include_bias=True)),
       ('ridge', Ridge(fit_intercept=False))
   ])
   # Intercept won't be penalized by regularization
   ```

3. **Algorithms without `intercept_` attribute** (e.g., some custom estimators)

### Use `include_bias=False` when:

1. **Standard polynomial fitting** (default `fit_intercept=True`)
   ```python
   model = Pipeline([
       ('poly', PolynomialFeatures(2, include_bias=False)),
       ('linear', LinearRegression())  # fit_intercept=True by default
   ])
   # Clean separation: coef for polynomial terms, intercept_ for constant
   ```

2. **You want clear coefficient interpretation**
   - `coef_` → polynomial term coefficients only
   - `intercept_` → constant term (c₀)

## Example: Lane Detection with RANSAC

### Problem with `include_bias=True`
```python
model = make_pipeline(
    PolynomialFeatures(2, include_bias=True),  # Features: [1, y, y²]
    RANSACRegressor()
)
model.fit(X, y)

ransac = model.named_steps['ransacregressor']
coef = ransac.estimator_.coef_        # [c_bias, c₁, c₂]
intercept = ransac.estimator_.intercept_

# Confusion: which is the constant?
# Actual constant = coef[0] + intercept
constant_term = coef[0] + intercept    # ← Must combine them!
poly_coef = coef[1:]                   # [c₁, c₂]
```

### Solution with `include_bias=False`
```python
model = make_pipeline(
    PolynomialFeatures(2, include_bias=False),  # Features: [y, y²]
    RANSACRegressor()
)
model.fit(X, y)

ransac = model.named_steps['ransacregressor']
coef = ransac.estimator_.coef_        # [c₁, c₂] ← Clean!
intercept = ransac.estimator_.intercept_  # c₀

# Convert to numpy polyfit format: [c₂, c₁, c₀]
fit_coefficients = np.concatenate([coef[::-1], [intercept]])
```

## Coefficient Mapping

For a 2nd degree polynomial `x = c₂·y² + c₁·y + c₀`:

| Setup | `coef_` | `intercept_` | Actual c₀ | Conversion |
|-------|---------|--------------|-----------|------------|
| `include_bias=True` | `[c_bias, c₁, c₂]` | intercept | `c_bias + intercept` | `[coef[2], coef[1], coef[0] + intercept]` |
| `include_bias=False` | `[c₁, c₂]` | c₀ | `intercept` | `[coef[1], coef[0], intercept]` |

## Recommendation for Your Use Case

**Use `include_bias=False`** for polynomial curve fitting because:
- ✅ Clearer coefficient interpretation
- ✅ No confusion about constant terms
- ✅ Matches `np.polyfit` convention directly
- ✅ Easier debugging

```python
def central_curve_fit_ransac(mask: np.ndarray, polynomial_order: int = 2) -> np.ndarray:
    # ...existing code...
    
    model = make_pipeline(
        PolynomialFeatures(polynomial_order, include_bias=False),  # ← Recommended
        RANSACRegressor(...)
    )
    model.fit(X, y)
    
    ransac = model.named_steps['ransacregressor']
    coef = ransac.estimator_.coef_        # [c₁, c₂, ..., cₙ]
    intercept = ransac.estimator_.intercept_  # c₀
    
    # Convert to numpy format: [cₙ, ..., c₂, c₁, c₀]
    fit_coefficients = np.concatenate([coef[::-1], [intercept]])
    
    return fit_coefficients
```

## Summary

- `include_bias=True` → Adds constant feature, useful for no-intercept models
- `include_bias=False` → No constant feature, cleaner for standard regression
- **For polynomial fitting with RANSAC: use `include_bias=False`**