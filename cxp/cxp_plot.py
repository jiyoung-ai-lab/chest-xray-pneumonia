# ============================================================
# cxp_plot.py
# ============================================================

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# Training History
# ============================================================

def _add_history(
    fig,
    history,
    run,
    col,
    metric,
):
    """
    Add train/validation history for one run.
    """

    h = history[
        history["run"] == run
    ]

    if h.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=h["epoch"],
            y=h[f"train_{metric}"],
            mode="lines",
            name=run,
            legendgroup=run,
            line=dict(
                dash="dot",
                width=2,
            ),
        ),
        row=1,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=h["epoch"],
            y=h[f"validation_{metric}"],
            mode="lines",
            name=f"{run} Validation",
            legendgroup=run,
            showlegend=False,
            line=dict(
                width=3,
            ),
        ),
        row=1,
        col=col,
    )


def plot_experiment_history(
    history,
    experiment_name,
    metric="loss",
):
    """
    Plot training history for all runs in an experiment.

    For experiments with a 20-epoch extension, runs are
    displayed according to their epoch length.
    """

    runs = (
        history["run"]
        .drop_duplicates()
        .tolist()
    )

    short_runs = []
    long_runs = []

    for run in runs:

        run_history = history[
            history["run"] == run
        ]

        max_epoch = run_history["epoch"].max()

        if max_epoch <= 10:
            short_runs.append(run)
        else:
            long_runs.append(run)

    # --------------------------------------------------------
    # Create subplot
    # --------------------------------------------------------

    if long_runs:

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Initial Runs",
                "Extended Runs",
            ],
            horizontal_spacing=0.10,
        )

        for run in short_runs:
            _add_history(
                fig=fig,
                history=history,
                run=run,
                col=1,
                metric=metric,
            )

        for run in long_runs:
            _add_history(
                fig=fig,
                history=history,
                run=run,
                col=2,
                metric=metric,
            )

    else:

        fig = make_subplots(
            rows=1,
            cols=1,
        )

        for run in short_runs:
            _add_history(
                fig=fig,
                history=history,
                run=run,
                col=1,
                metric=metric,
            )

    ylabel = (
        "Loss"
        if metric == "loss"
        else "Accuracy"
    )

    fig.update_xaxes(
        title_text="Epoch"
    )

    fig.update_yaxes(
        title_text=ylabel
    )

    fig.update_layout(
        title=f"{experiment_name} — {ylabel}",
        width=1100,
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.show()


# ============================================================
# Validation Metric Comparison
# ============================================================

def plot_metric_comparison(
    results,
    metrics=None,
    title="Validation Metric Comparison",
):
    """
    Compare validation metrics across experiment runs.
    """

    if metrics is None:
        metrics = [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
        ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in results.columns
    ]

    fig = go.Figure()

    for metric in available_metrics:

        fig.add_trace(
            go.Bar(
                x=results["run"],
                y=results[metric],
                name=metric.replace("_", " ").title(),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Run",
        yaxis_title="Score",
        barmode="group",
        yaxis=dict(
            range=[0, 1],
        ),
        width=1100,
        height=550,
        hovermode="x unified",
    )

    fig.show()


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(
    y_true,
    y_prob,
    title="ROC Curve",
):
    """
    Plot ROC curve for one model/run.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC-AUC = {roc_auc:.4f}",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(
                dash="dash"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=600,
    )

    fig.show()


# ============================================================
# Precision-Recall Curve
# ============================================================

def plot_pr_curve(
    y_true,
    y_prob,
    title="Precision-Recall Curve",
):
    """
    Plot Precision-Recall curve for one model/run.
    """

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_prob,
        )
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR-AUC = {pr_auc:.4f}",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=800,
        height=600,
    )

    fig.show()


# ============================================================
# Error Images
# ============================================================

def show_error_images(
    dataframe,
    data_dir,
    idx_to_class,
    n=8,
    title="Error Analysis",
):
    """
    Display selected error images.
    """

    sample_df = dataframe.head(n)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        sample_df.iterrows(),
    ):

        image_path = (
            data_dir
            / row["Folder"]
            / row["File"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        ax.imshow(
            image,
            cmap="gray",
        )

        ax.set_title(
            f"Actual: {row['Folder']}\n"
            f"Pred: {idx_to_class[row['y_pred']]}\n"
            f"P(PNEU): {row['y_prob']:.3f}"
        )

        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# Confusion Matrix Comparison
# ============================================================

def plot_confusion_matrix_comparison(
    results,
    title="Validation Confusion Matrix",
):
    """
    Compare confusion matrix counts across experiment runs.
    """

    fig = make_subplots(
        rows=1,
        cols=len(results),
        subplot_titles=results["run"].tolist(),
    )

    for col, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        z = [
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]],
        ]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=["NORMAL", "PNEUMONIA"],
                y=["NORMAL", "PNEUMONIA"],
                text=z,
                texttemplate="%{text}",
                showscale=False,
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        title=title,
        width=max(900, 300 * len(results)),
        height=400,
    )

    fig.update_xaxes(
        title_text="Predicted",
    )

    fig.update_yaxes(
        title_text="Actual",
    )

    fig.show()

# ============================================================
# ROC Curve Comparison
# ============================================================

def plot_roc_curve_comparison(
    prediction_results,
    title="Validation ROC Curve Comparison",
):
    """
    Compare ROC curves across experiment runs.
    """

    fig = go.Figure()

    for run, result in prediction_results.items():

        y_true = result["labels"]
        y_prob = result["probabilities"]

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob,
        )

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{run} (AUC={roc_auc:.4f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(
                dash="dash",
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=850,
        height=600,
    )

    fig.show()

# ============================================================
# Precision-Recall Curve Comparison
# ============================================================

def plot_pr_curve_comparison(
    prediction_results,
    title="Validation Precision-Recall Curve Comparison",
):
    """
    Compare Precision-Recall curves across experiment runs.
    """

    fig = go.Figure()

    for run, result in prediction_results.items():

        y_true = result["labels"]
        y_prob = result["probabilities"]

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                y_prob,
            )
        )

        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{run} (PR-AUC={pr_auc:.4f})",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=850,
        height=600,
    )

    fig.show()