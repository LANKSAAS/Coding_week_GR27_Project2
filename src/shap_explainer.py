"""
shap_explainer.py — SHAP-based model explainability utilities.

Provides helpers to compute SHAP values using ``TreeExplainer`` and to
generate summary, feature-importance, and per-prediction waterfall plots.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np


def get_shap_explainer(model, X_background):
    """Create a SHAP TreeExplainer.

    Parameters
    ----------
    model : tree-based estimator
        A fitted model compatible with ``shap.TreeExplainer``.
    X_background : pd.DataFrame or np.ndarray
        A representative background dataset (e.g. a sample of the
        training set) used for computing expected values.

    Returns
    -------
    shap.TreeExplainer
    """
    return shap.TreeExplainer(model, X_background)


def get_shap_values(explainer, X):
    """Compute SHAP values for the given data.

    Returns
    -------
    shap.Explanation
    """
    return explainer(X)


def plot_summary(shap_values, feature_names=None, max_display=16, show=True):
    """Global beeswarm summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if show:
        plt.show()
    return plt.gcf()


def plot_feature_importance(shap_values, feature_names=None, max_display=16, show=True):
    """Mean |SHAP| bar plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if show:
        plt.show()
    return plt.gcf()


def plot_waterfall(shap_values, idx: int = 0, max_display: int = 14, show: bool = True):
    """Single-prediction waterfall plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        Full SHAP explanation object (multi- or single-output).
    idx : int
        Row index of the instance to explain.
    max_display : int
        Number of features to display.
    show : bool
        Whether to call ``plt.show()``.
    """
    plt.figure(figsize=(10, 6))
    # For multi-class, shap_values is 3-D; pick the predicted class
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        predicted_class = int(np.argmax(shap_values.values[idx].sum(axis=0)))
        sv = shap_values[idx, :, predicted_class]
    else:
        sv = shap_values[idx]

    shap.plots.waterfall(sv, max_display=max_display, show=False)
    plt.tight_layout()
    if show:
        plt.show()
    return plt.gcf()