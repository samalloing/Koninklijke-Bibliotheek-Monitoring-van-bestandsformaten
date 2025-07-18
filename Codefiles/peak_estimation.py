from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from pandas.tseries.offsets import DateOffset
from scipy.stats.mstats import ttest_1samp
from sklearn.pipeline import make_pipeline
from numpy.polynomial import Polynomial
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from itertools import product
from math import sqrt
import pandas as pd
import numpy as np

window_size = 8
significance_level = 0.05
params=None
minimum_k = 3 
minimum_h = 1
k_range=range(minimum_k, 8, 1) 
h_range=np.linspace(minimum_h, 3, 10)

def last_argmax(arr):
    # Return last index of maximum value
    return len(arr) - np.argmax(arr[::-1]) - 1

# Custom scoring function
def argmax_index_diff_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If y_true or y_pred is multidimensional, reduce to 1D
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    peak_penalty = abs(last_argmax(y_true) - last_argmax(y_pred))
    # Fit shape penalty (use MSE)
    shape_penalty = root_mean_squared_error(y_true, y_pred)
    return -0.8 * peak_penalty - 0.2 * shape_penalty
    # Calculate absolute index difference between argmax
    # return -abs(last_argmax(y_true) - last_argmax(y_pred)) # negative because GridSearchCV maximizes score

# Create the scorer
custom_scorer = make_scorer(argmax_index_diff_score, greater_is_better=True)

def s_core(x, y, scaler, left_bound, score_func, k_range, h_range):
    best_score, best_params, best_peak, best_li, best_ri, best_y = np.inf, None, None, None, None, None
    y_scaled = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    for k, h in product(k_range, h_range):
        scores = np.zeros_like(y_scaled)        
        n = len(y_scaled)
        for i in range(k, n - k):
            scores[i] = score_func(y_scaled, i, k)
        positive_scores = scores[scores > 0]
        if len(positive_scores) == 0:
            continue
        m_, s_ = np.mean(positive_scores), np.std(positive_scores)
        thresholded = (scores > 0) & (scores - m_ >= h * s_)
        # print(np.concatenate(([(scores >= 0)], [(scores - m_ > h * s_)]), axis=0), h * s_, scores - m_)
        if not np.any(thresholded):
            continue
        
        peak_idx = last_argmax(y_scaled * thresholded)
        # peak_idx = np.argmax(y_scaled * thresholded)
        # y_pred = np.zeros_like(y_scaled)
        # y_pred[peak_idx] = y_scaled[peak_idx]
        score = abs(peak_idx - last_argmax(y))
        # rmse = root_mean_squared_error(y_scaled, y_pred)
        if score < best_score:
            best_score = score # best_rmse = rmse
            best_params = (int(k), h)
            best_peak = peak_idx
            best_y = y_scaled
            d2y = np.gradient(np.gradient(y_scaled))
            li_candidates = [i for i in range(1, peak_idx) if np.sign(d2y[i]) != np.sign(d2y[i - 1])]
            ri_candidates = [i for i in range(peak_idx + 1, n - 1) if np.sign(d2y[i]) != np.sign(d2y[i + 1])]
            best_li = left_bound + max(li_candidates) if li_candidates else None
            best_ri = left_bound + min(ri_candidates) if ri_candidates else None
    if best_y is None:
        return pd.DataFrame({'Count': y_scaled}, index=x), None, tuple(np.full(2, None).tolist()), tuple(np.full(3, np.nan).tolist())
    return pd.DataFrame({'Count': best_y}, index=x), best_peak, (best_li, best_ri), tuple([*best_params, np.nan])

# S1: max signed distances to neighbors
def s1(y, i, k):
    return 0.5 * (max(y[i] - y[i - k:i]) + max(y[i] - y[i + 1:i + k + 1]))

# S2: average signed distances to neighbors
def s2(y, i, k):
    return 0.5 * (np.mean(y[i] - y[i - k:i]) + np.mean(y[i] - y[i + 1:i + k + 1]))

# S3: signed distance from mean of window
def s3(y, i, k):
    neighbors = np.concatenate((y[i - k:i], y[i + 1:i + k + 1]))
    return y[i] - np.mean(neighbors)

# Wrapper functions
def s_1(x, y, scaler, left_bound, k_range=k_range, h_range=h_range):
    return s_core(x, y, scaler, left_bound, s1, k_range, h_range)

def s_2(x, y, scaler, left_bound, k_range=k_range, h_range=h_range):
    return s_core(x, y, scaler, left_bound, s2, k_range, h_range)

def s_3(x, y, scaler, left_bound, params=None, k_range=k_range, h_range=h_range):
    return s_core(x, y, scaler, left_bound, s3, k_range, h_range)

# Minimax objective function (minimizing the maximum absolute error)
# def minimax_objective(params, t, y_true):
#     m, p, q = params
#     y_pred = bass_model(t, m, p, q)
#     return y_true - y_pred

def window_avg_zero(window, population_mean = 0):
    pvalue = ttest_1samp(window, population_mean, alternative='less').pvalue
    if np.isnan(pvalue):
        return -np.inf
    else:
        return pvalue

def array_bound(y, side, window_size, significance_level = significance_level):
    array_end_index = len(y) - 1
    window_size = min(array_end_index - (array_end_index + 1) % 2 - 1 , window_size)
    y_mean = np.mean(y[np.nonzero(y)])
    if side == 'left':
        right = window_size
        left = 0
        window = y[left:right]
        pvalue = window_avg_zero(window, y_mean)  
        while pvalue < significance_level and right != array_end_index: 
            left = left + 1 
            right = right + 1 
            window = y[left:right] 
            pvalue = window_avg_zero(window, y_mean)
        if pvalue >= significance_level:
            return left
        else:
            # print('unbounded left')
            return 0
    if side == 'right':
        right = array_end_index
        left = right - window_size
        window = y[left:right]
        pvalue = window_avg_zero(window, y_mean)
        while pvalue < significance_level and left != 0:
            left = left - 1
            right = right - 1
            window = y[left:right]
            pvalue = window_avg_zero(window, y_mean)
        if pvalue >= significance_level:
            return right
        else:
            # print('unbounded right')
            return array_end_index

# Function to fit Bass model using Minimax optimization
# def fit_bass_minimax(t, y, scaler, left_bound, params=None):
#     max_y = np.max(y)
#     param_tuple = None
#     if params is None:
#         result = least_squares(minimax_objective, x0=[max_y * 32, 0.03, 0.38], args=(t, y), method="lm")
#         params = result.x
#         param_tuple = tuple([*params])
#     result_array = scaler.inverse_transform(bass_model(t,*params).reshape(-1,1)).flatten() 
#     max_result_array = np.max(result_array)
#     dy = np.gradient(result_array)
#     peak_idx = np.argmax(result_array)
#     left = left_bound + np.argmax(dy[:peak_idx]) if len(np.unique_counts(dy[:peak_idx]).values) > 1 else None
#     right = left_bound + peak_idx + np.argmin(dy[peak_idx:]) if len(np.unique_counts(dy[peak_idx:]).values) > 1 else None
#     return pd.DataFrame({'pol_index' : t, 'Count' : result_array}).set_index('pol_index'), max_result_array, (left, right), param_tuple

def bass_model(t, m, p, q):
    exp_term = np.exp(-(p + q) * t)
    return m * (((p + q) ** 2) / p) * (exp_term / ((1 + (q / p) * exp_term) ** 2))

def hybrid_objective_bass(params, x, y):
    y_pred = bass_model(x, *params)
    # Peak alignment
    peak_penalty = abs(last_argmax(y) - round(np.log(params[2]/params[1]) / (params[1] + params[2])))
    # Fit shape penalty (use MSE)
    shape_penalty = root_mean_squared_error(y, y_pred)
    return 0.8 * peak_penalty + 0.2 * shape_penalty

def fit_bass_minimax(t, y, scaler, left_bound, params=None):  
    if params is None:
        x = np.arange(0,len(y)) #np.linspace(0,1, num = len(y)) #
        max_y = np.max(y)
        initial_guess = [max_y * 32, 0.03, 0.38] # [0.03, 0.38]
        # Optional bounds for physical plausibility
        default_bounds = [(1, max_y * 100), (1e-5, 1.0), (1e-5, 1.0)] # (1, max_y * 100) (0.01, 0.03), (0.3, 0.5)
        result = minimize(hybrid_objective_bass, 
                          x0=initial_guess,
                          args=(x, y),
                          bounds=default_bounds,
                          method='L-BFGS-B', # 'Nelder-Mead'
                          options={'maxiter': 1000, # 'Powell'
                                   'ftol': 1e-10,
                                   'gtol': 1e-8,
                                   'eps': 1e-7,  # Encourage larger early steps
                                #    'disp': True  # Show progress if helpful
                                }) 
        params = result.x
        m, p, q = params
        result_array = scaler.inverse_transform(bass_model(x, *params).reshape(-1, 1)).flatten() # t
        peak_idx = round(np.log(q/p) / (p + q)) # last_argmax(result_array)
        ddy = np.gradient(np.gradient(result_array))
        # Identify sign changes in second derivative
        sign_changes = np.where(np.diff(np.sign(ddy)) != 0)[0]
        left_candidates = sign_changes[sign_changes < peak_idx]
        right_candidates = sign_changes[sign_changes > peak_idx]
        del sign_changes
        discriminant = np.log(2 + sqrt(3))/(p+q)
        left = left_bound + round(peak_idx - discriminant) if len(left_candidates) > 0 else np.nan # 
        right = left_bound + round(peak_idx + discriminant) if len(right_candidates) > 0 else np.nan # 
    else:
        x = t - t[0]
        result_array = scaler.inverse_transform(bass_model(x, *params).reshape(-1, 1)).flatten()
        # print('Bass t, t[0]:', t, t[0])
        dy = np.gradient(result_array)
        peak_indices = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
        if len(peak_indices):
            highest_peak_idx = peak_indices[last_argmax(result_array[peak_indices])]
            peak_idx = x[highest_peak_idx]
        else:
            peak_idx = np.nan
        ddy = np.gradient(dy)
        # Identify sign changes in second derivative
        sign_changes = np.where(np.diff(np.sign(ddy)) != 0)[0]
        left_candidates = sign_changes[sign_changes < peak_idx]
        right_candidates = sign_changes[sign_changes > peak_idx]
        del sign_changes
        left = left_bound + t[max(left_candidates)] if len(left_candidates) > 0 else np.nan
        right = left_bound + t[min(right_candidates)] if len(right_candidates) > 0 else np.nan
    return pd.DataFrame({'pol_index': t, 'Count': result_array}).set_index('pol_index'), peak_idx, (left, right), params

def sine_function(x, A, B, C, D = 0):
    return A * np.sin(B * x + C) + D

def argmax_index_diff_objective_bc(params_bc, x, y_true):
    B, C = params_bc
    # Temporary guess for A and D just to get a shape for argmax
    A_guess = 1.0
    y_pred = sine_function(x, A_guess, B, C)
    if np.all(y_pred == 0):
        return np.inf
    return abs(last_argmax(y_true) - last_argmax(y_pred))

def hybrid_objective_sine(params, x, y):
    y_pred = sine_function(x, *params)

    # Peak alignment
    d1 = np.gradient(y_pred)
    # Check for mathematical bottoms (local minima)
    bottom_indices = np.where((d1[:-1] < 0) & (d1[1:] > 0))[0] + 1
    if len(bottom_indices) > 0:
        return np.inf  # Disallow bottoms entirely

    peak_indices = np.where((d1[:-1] > 0) & (d1[1:] < 0))[0] + 1    
    if len(peak_indices):
        highest_peak_idx = peak_indices[last_argmax(y_pred[peak_indices])]
        peak_idx = x[highest_peak_idx]
    else:
        peak_idx = np.nan
    if peak_idx:
        peak_penalty = abs(last_argmax(y) - peak_idx)
        # print(f'A = {params[0]}, B = {params[1]}, C = {params[2]}, penalty = {0.8 * peak_penalty + 0.2 * root_mean_squared_error(y, y_pred)}')
    else:
        return np.inf
    # Fit shape penalty (use MSE)
    shape_penalty = root_mean_squared_error(y, y_pred)
    return 0.8 * peak_penalty + 0.2 * shape_penalty

from scipy.optimize import approx_fprime

def custom_jac(x, *args):
    steps = np.array([1e-5, 1e-5, 1.0])  # eps for A, B, C
    return approx_fprime(x, hybrid_objective_sine, steps, *args)

def fit_sine_minimax(t, y, scaler, left_bound, params_abc=None):
    # KB: eps
    # DANS: cmd, tar, prj
    # All: pptx, xlsx, mp3, htm
    # Step 1: Fit B and C to minimize index difference
    if not params_abc:
        x = np.arange(len(y))
        L = len(x)
        A_bounds = (1, np.max(y)) # (1e-3, max_y)
        B_bounds = (np.pi / L, 2 * np.pi / L)    # (np.pi / (2 * L), np.pi / L)
        C_bounds = (-2 * np.pi, 2 * np.pi)
        result = minimize(hybrid_objective_sine,
                          x0=[(np.max(y) - np.min(y)) / 2, (B_bounds[0] + B_bounds[1]) / 2 , np.pi / L], # (1.0, 0.0) Middle of B range, neutral phase
                          args=(x, y),
                          bounds=[A_bounds, B_bounds, C_bounds],
                          method='Nelder-Mead', #'Powell' 'L-BFGS-B' 
                        #   jac=custom_jac,
                          options={'maxiter': 1000,
                                   'gtol': 1e-8,
                                   'ftol': 1e-10,
                                   'eps': 1e-7,
                                #    'disp': True
                                })
        A, B, C = result.x
        param_tuple = tuple(map(float, (A, B, C)))  # Ensure standard Python floats
    else:
        x = t - t[0]
        # print('Sine t, t[0]:', t, t[0])
        A, B, C = params_abc
        param_tuple = (A, B, C)
    result_array = scaler.inverse_transform(sine_function(x, A, B, C).reshape(-1, 1)).flatten()
    # Compute peak via algebraic method (1st & 2nd derivative)
    d1 = np.gradient(result_array)
    d2 = np.gradient(d1)
    peak_indices = np.where((d1[:-1] > 0) & (d1[1:] < 0))[0] + 1
    # print('peak_indices', peak_indices)
    if len(peak_indices):
        highest_peak_idx = peak_indices[last_argmax(result_array[peak_indices])]
        peak_idx = x[highest_peak_idx]
    else:
        peak_idx = np.nan

    # # Identify sign changes in second derivative
    # sign_changes = np.where(np.diff(np.sign(d2)) != 0)[0]
    # # print('sign_changes', sign_changes)
    # left_candidates = sign_changes[sign_changes < peak_idx]
    # right_candidates = sign_changes[sign_changes > peak_idx]
    # del sign_changes
    # left = t[max(left_candidates)] if left_candidates.size > 0 else np.nan # left_bound +
    # right = t[min(right_candidates)] if right_candidates.size > 0 else np.nan # left_bound +
    left = right = None
    if not np.isnan(peak_idx):
        peak_idx = int(np.where(x == peak_idx)[0][0])  # ensure index not value
        # Left: search backward for sign change
        for i in range(peak_idx - 1, 0, -1):
            if np.sign(d2[i]) != np.sign(d2[i - 1]):
                left = left_bound + i
                break
        # Right: search forward for sign change
        for i in range(peak_idx + 1, len(d2) - 1):
            if np.sign(d2[i]) != np.sign(d2[i + 1]):
                right = left_bound + i
                break

    # print('left, right, first, last', left, right, t[0], t[-1])
    return pd.DataFrame({'pol_index': t, 'Count': result_array}).set_index('pol_index'), peak_idx, (left, right), param_tuple

# Function to check for unique global peak
def has_single_global_peak(y_vals):
    dy = np.gradient(y_vals)
    peak_indices = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
    return np.count_nonzero(y_vals == np.max(y_vals)) == 1 and len(peak_indices) > 0

def polynomial(x, y, scaler, left_bound, params=None, scoring=custom_scorer):
    param_tuple = None

    if params is None:
        x_range = np.arange(len(y))
        grid_search = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(), Ridge()),
                                param_grid={'polynomialfeatures__degree': np.arange(2,4), # https://github.com/dipenpandit/degree-selection-for-polynomial-regression/blob/main/housing%20price%20model%20evaluation.ipynb
                                            'ridge__alpha': np.logspace(-6, 6, 13)}, # -4,3,10 # Niet meer dan tien waarden https://archive.is/jbh19
                                scoring=scoring,
                                n_jobs=-1)
        grid_search.fit(x_range.reshape(-1, 1), np.ravel(y))

        candidates = []
        for p in grid_search.cv_results_['params']:
            model = make_pipeline(PolynomialFeatures(degree=p['polynomialfeatures__degree']),
                                  Ridge(alpha=p['ridge__alpha']))
            model.fit(x_range.reshape(-1, 1), np.ravel(y))
            y_pred = scaler.inverse_transform(model.predict(x_range.reshape(-1, 1)).reshape(-1, 1)).flatten()
            score = argmax_index_diff_score(y, y_pred)
            candidates.append({'model' : model,
                               'params' : p,
                               'score' : score,
                               'y_pred' : y_pred})
        one_peak_models = [c for c in candidates if has_single_global_peak(c['y_pred'])]
        best = max(one_peak_models, key=lambda c: c['score']) if one_peak_models else max(candidates, key=lambda c: c['score'])
        model = best['model']
        y_pred = best['y_pred']
        best_params = best['params']
        param_tuple = (best_params['polynomialfeatures__degree'], best_params['ridge__alpha'], np.nan)
    else:
        x = x - x[0]
        model = make_pipeline(PolynomialFeatures(degree=int(params[0])),
                              Ridge(alpha=params[1]))
        model.fit(x.reshape(-1, 1), np.ravel(y))
        y_pred = scaler.inverse_transform(model.predict(x.reshape(-1, 1)).reshape(-1, 1)).flatten()

    # Extract polynomial coefficients
    # poly = model.named_steps['polynomialfeatures']
    # ridge = model.named_steps['ridge']
    # degree = poly.degree
    # coefs = np.zeros(degree + 1)
    # coefs[:len(ridge.coef_)] = ridge.coef_
    # coefs[0] = ridge.intercept_

    # # Build polynomial and compute first and second derivatives
    # p = Polynomial(coefs)
    # d1 = p.deriv()
    # d2 = d1.deriv()

    # # Find critical points (first derivative = 0)
    # crit_roots = d1.roots()
    # crit_roots = [r.real for r in crit_roots if np.isreal(r) and np.min(x) <= r <= np.max(x)]
    # crit_vals = [(p(r)).item() for r in crit_roots]
    # if crit_vals:
    #     peak_val = max(crit_vals)
    #     peak_root = crit_roots[crit_vals.index(peak_val)]
    #     peak_idx = int(round(peak_root))
    # else:
    #     peak_idx = np.nan
    # # Second derivative roots for inflection bounds
    # inflect_roots = d2.roots()
    # inflect_roots = [r.real for r in inflect_roots if np.isreal(r) and np.min(x) <= r <= np.max(x)]
    # print('Polynoom inflect_roots', inflect_roots)
    # inflect_idx = sorted(set(int(round(r)) for r in inflect_roots))
    # left = left_bound + max([i for i in inflect_idx if i < peak_idx], default=None) if inflect_idx else np.nan #  
    # right = left_bound + min([i for i in inflect_idx if i > peak_idx], default=None) if inflect_idx else np.nan # 
    # Compute peak via algebraic method (1st & 2nd derivative)
    d1 = np.gradient(y_pred)
    d2 = np.gradient(d1)
    peak_indices = np.where((d1[:-1] > 0) & (d1[1:] < 0))[0] + 1
    if len(peak_indices):
        highest_peak_idx = peak_indices[last_argmax(y_pred[peak_indices])]
        peak_idx = x[highest_peak_idx]
    else:
        peak_idx = np.nan

    # Estimate left/right inflection bounds (if they exist)
    # l = [i for i in np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0] if i < peak_idx]
    # r = [i for i in np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0] if i > peak_idx]
    # left = left_bound + max(l) if l else None 
    # right = left_bound + min(r) if r else None
    left = right = None
    if not np.isnan(peak_idx):
        peak_idx = int(np.where(x == peak_idx)[0][0])  # ensure index not value
        # Left: search backward for sign change
        for i in range(peak_idx - 1, 0, -1):
            if np.sign(d2[i]) != np.sign(d2[i - 1]):
                left = left_bound + i
                break
        # Right: search forward for sign change
        for i in range(peak_idx + 1, len(d2) - 1):
            if np.sign(d2[i]) != np.sign(d2[i + 1]):
                right = left_bound + i
                break

    # print('Polynoom left, right, first, last', left, right, x[0], x[-1])
    return pd.DataFrame({'pol_index': x.flatten(), 'Count': y_pred}).set_index('pol_index'), peak_idx, (left, right), param_tuple

# Main spline model selection and fitting
def spline(x, y, scaler, left_bound, params=None, scoring=custom_scorer):
    param_tuple = None

    if params is None:
        x_range = np.arange(len(y))
        g=GridSearchCV(make_pipeline(SplineTransformer(),Ridge()),{'splinetransformer__degree' : np.arange(2,4), # https://github.com/dipenpandit/degree-selection-for-polynomial-regression/blob/main/housing%20price%20model%20evaluation.ipynb
                                                                   'splinetransformer__n_knots' : np.arange(2,6), # https://www.sfu.ca/sasdoc/sashtml/stat/chap65/sect30.htm
                                                                   'ridge__alpha' : np.logspace(-6, 6, 13)}, # -4,3,10
                                                                   scoring=scoring,
                                                                   n_jobs=-1) # 'neg_max_error'
        g.fit(x_range.reshape(-1, 1), np.ravel(y))
        candidates = []

        # Evaluate each model for global peak condition
        for i, params in enumerate(g.cv_results_['params']):
            model = make_pipeline(SplineTransformer(degree=params['splinetransformer__degree'],
                                                    n_knots=params['splinetransformer__n_knots']),
                                  Ridge(alpha=params['ridge__alpha']))
            model.fit(x_range.reshape(-1, 1), np.ravel(y))
            y_pred = scaler.inverse_transform(model.predict(x_range.reshape(-1, 1)).reshape(-1, 1)).flatten()
            score = argmax_index_diff_score(y, y_pred)
            candidates.append({'model': model,
                               'params': params,
                               'score': score,
                               'y_pred': y_pred})
        # Separate into models with one global peak and fallback
        one_peak_models = [c for c in candidates if has_single_global_peak(c['y_pred'])]
        best = max(one_peak_models, key=lambda c: c['score']) if one_peak_models else max(candidates, key=lambda c: c['score'])
        model = best['model']
        y_pred = best['y_pred']
        best_params = best['params']
        param_tuple = (best_params['splinetransformer__degree'],
                       best_params['splinetransformer__n_knots'],
                       best_params['ridge__alpha'])
    else:
        # Manual model
        x = x - x[0]
        model = make_pipeline(SplineTransformer(degree=int(params[0]), n_knots=int(params[1])),
                              Ridge(alpha=params[2]))
        model.fit(x.reshape(-1, 1), np.ravel(y))
        y_pred = scaler.inverse_transform(model.predict(x.reshape(-1, 1)).reshape(-1, 1)).flatten()

    # Compute peak via algebraic method (1st & 2nd derivative)
    d1 = np.gradient(y_pred)
    d2 = np.gradient(d1)
    peak_indices = np.where((d1[:-1] > 0) & (d1[1:] < 0))[0] + 1
    if len(peak_indices):
        highest_peak_idx = peak_indices[last_argmax(y_pred[peak_indices])]
        peak_index = x[highest_peak_idx]
    else:
        peak_index = np.nan
    # Estimate left/right inflection bounds (if they exist)
    # l = [i for i in np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0] if i < peak_index]
    # r = [i for i in np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0] if i > peak_index]
    # li = left_bound + max(l) if l else None # 
    # ri = left_bound + min(r) if r else None # 
    li = ri = None
    if not np.isnan(peak_index):
        peak_idx = int(np.where(x == peak_index)[0][0])  # ensure index not value
        # Left: search backward for sign change
        for i in range(peak_idx - 1, 0, -1):
            if np.sign(d2[i]) != np.sign(d2[i - 1]):
                li = left_bound + i
                break
        # Right: search forward for sign change
        for i in range(peak_idx + 1, len(d2) - 1):
            if np.sign(d2[i]) != np.sign(d2[i + 1]):
                ri = left_bound + i
                break

    return pd.DataFrame(y_pred, columns=['Count'], index=x), peak_index, (li, ri), param_tuple

def savitzky_golay(x, y, scaler, left_bound, window_range = window_size, poly_order=3):
    w=min(window_range,len(y));f=savgol_filter(y,w,poly_order);d1=np.gradient(f);d2=np.gradient(d1)
    pks=np.where((d1[:-1]>0)&(d1[1:]<0))[0];p=last_argmax(f);li=ri=None # np.argmax(f) 
    left=[i for i in np.where(np.sign(d2[:-1])!=np.sign(d2[1:]))[0] if i<p]
    right=[i for i in np.where(np.sign(d2[:-1])!=np.sign(d2[1:]))[0] if i>p]
    
    if left:
        li= left_bound + max(left)
    if right:
        ri= left_bound + min(right)
    return pd.DataFrame(scaler.inverse_transform(f.reshape(-1,1)).flatten(),columns=['Count'],index=x), p ,(li,ri), tuple(np.full(3, np.nan).tolist())

def reorient_frame(bounded, frame, peak, left, right):
    if left:
        if left >= bounded.index[0]:
            left = bounded.loc[(bounded.index == int(left)), 'Date'].values[0]
        else:
            try:
                left = bounded.iloc[0]['Date'] + DateOffset(months=int(left-bounded.index[0]))
            except:
                # print('Date offset left out of bounds', left, bounded.index[0], bounded.index[-1])
                left = np.nan
    if right:
        if right <= bounded.index[-1]:
            right = bounded.loc[(bounded.index == int(right)), 'Date'].values[0]
        else:
            try:
                right = bounded.iloc[-1]['Date'] + DateOffset(months=int(right-bounded.index[-1]))
            except:
                # print('Date offset right out of bounds', right, bounded.index[0], bounded.index[-1])
                right = np.nan
    reorient = bounded.merge(frame, left_index=True, right_index=True)
    inflection_points = (left, right)
    # print('reorient', reorient, 'peak', peak)
    if not np.isnan(peak) and peak >= reorient.index.values[0] and peak <= reorient.index.values[-1]:
        max_index = reorient.loc[reorient.index == peak, 'Date'].values[-1]
    else:
        max_index = np.nan
    reorient = reorient.reset_index(drop=True).set_index('Date')
    max_value = reorient['Count'].max()
    # if reorient['Count'].value_counts().get(max_value, 0) == 1:
        # max_index = reorient.loc[reorient['Count'] == max_value].index.values[-1]

    # if max_index != reorient['Count'].index.values[0] and max_index != reorient['Count'].index.values[-1]:
    #     return reorient, inflection_points, max_index, max_value
    return reorient, inflection_points, max_index, max_value
    # return reorient, inflection_points, None, None

def frame_bound(rep):
    rep['pol_index'] = np.arange(len(rep), dtype=np.int64).reshape(-1,1)
    left_bound = array_bound(rep['Count'].values, 'left', window_size)
    right_bound = array_bound(rep['Count'].values, 'right', window_size)
    rep = rep.iloc[left_bound:right_bound]
    scaler = StandardScaler()
    rep['Count'] = scaler.fit_transform(rep['Count'].values.reshape(-1, 1))
    rep['Date'] = rep.index
    rep.reset_index(inplace=True)
    rep = rep[['pol_index', 'Date', 'Count']]
    rep.set_index('pol_index', inplace=True)
    return rep, scaler, left_bound

def bootstrap_reduce_repo(unfiltered_repo, target_frac=0.632):
    repo = unfiltered_repo.copy()
    # Ensure counts are valid integers
    # original_counts = repo['Count'].fillna(0).astype(int).clip(lower=0).to_numpy()
    # total_original = original_counts.sum()
    # target_total = int(round(target_frac * total_original))
    # # Generate a long array of indices repeated by their count
    # expanded_indices = np.repeat(repo.index.to_numpy(), original_counts)
    # # Sample a subset of those indices
    # sampled_indices = np.random.choice(expanded_indices, size=target_total, replace=False)
    # # Count the frequency of each sampled index
    # sampled_counts = pd.Series(sampled_indices).value_counts().reindex(repo.index, fill_value=0)
    # # Assign back to DataFrame
    # repo['Count'] = sampled_counts.values
   
    # print('Bootstrapindex voor:', repo.index)
    max_value = repo['Count'].max()
    max_row = repo.loc[repo['Count']==max_value]
    repo = repo.loc[repo['Count']!=max_value].sample(frac=target_frac)
    repo = pd.concat([repo, max_row])
    repo.sort_index(inplace=True)
    # print('Bootstrapindex na:', repo.index)

    # repo.index = np.arange(len(repo), dtype=np.int64)
    return repo

def bootstrap_frame_bound(rep):
    rep['pol_index'] = np.arange(len(rep), dtype=np.int64).reshape(-1,1)
    left_bound = array_bound(rep['Count'].values, 'left', window_size)
    right_bound = array_bound(rep['Count'].values, 'right', window_size)
    rep = rep.iloc[left_bound:right_bound]
    rep = bootstrap_reduce_repo(rep)
    scaler = StandardScaler()
    rep['Count'] = scaler.fit_transform(rep['Count'].values.reshape(-1, 1))
    rep['Date'] = rep.index
    rep.reset_index(inplace=True)
    rep = rep[['pol_index', 'Date', 'Count']]
    rep.set_index('pol_index', inplace=True)
    return rep, scaler, left_bound