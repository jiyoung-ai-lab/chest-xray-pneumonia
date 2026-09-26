# ============================================================
# cxp_plot.py
# ============================================================
import math
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
    y_range=None,
):
    """
    Plot training history for all runs in an experiment.

    Existing LR analysis function.
    Kept based on run-level filtering.

    y_range:
        Optional Y-axis range, e.g. (0.8, 1.0).
        None keeps automatic range.
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

    if short_runs and long_runs:

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

        for run in runs:
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

    # --------------------------------------------------------
    # Optional Y-axis range
    # --------------------------------------------------------

    if y_range is not None:
        fig.update_yaxes(
            range=y_range
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
# Fine-Tuning Training History
# ============================================================

def plot_finetuning_history(
    history,
    experiment_runs,
    metric="loss",
    y_range=None,
):
    """
    Compare training history across selected fine-tuning runs.

    experiment_runs:
        [
            ("TL-RESNET50-FULL", "LR2-EPOCH20"),
            ("TL-RESNET50-PARTIAL", "LR2-EPOCH20"),
        ]

    Fine-Tuning analysis uses both experiment and run
    because the same run name can exist in multiple experiments.

    y_range:
        Optional Y-axis range, e.g. (0.8, 1.0).
        None keeps automatic range.
    """

    fig = go.Figure()

    for experiment, run in experiment_runs:

        h = history[
            (history["experiment"] == experiment)
            & (history["run"] == run)
        ].copy()

        if h.empty:
            continue

        label = experiment.replace(
            "TL-RESNET50-",
            "",
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=h["epoch"],
                y=h[f"train_{metric}"],
                mode="lines",
                name=f"{label} Train",
                legendgroup=label,
                line=dict(
                    dash="dot",
                    width=2,
                ),
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=h["epoch"],
                y=h[f"validation_{metric}"],
                mode="lines",
                name=f"{label} Validation",
                legendgroup=label,
                line=dict(
                    width=3,
                ),
            )
        )

    ylabel = (
        "Loss"
        if metric == "loss"
        else "Accuracy"
    )

    # --------------------------------------------------------
    # Optional Y-axis range
    # --------------------------------------------------------

    if y_range is not None:
        fig.update_yaxes(
            range=y_range
        )

    model_name = experiment_runs[0][0].replace("TL-", "").split("-")[0]
    
    fig.update_layout(
        title=f"{model_name} Fine-Tuning — {ylabel}",
        xaxis_title="Epoch",
        yaxis_title=ylabel,
        width=1100,
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
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
    score_range=(0, 1),
):
    """
    Compare validation metrics across experiment runs
    using a heatmap.

    Experiment + Run are displayed together because
    the same run name may exist in multiple experiments.

    score_range:
        Heatmap score range.
        Default: (0, 1)
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

    metric_labels = [
        metric.replace("_", " ").title()
        for metric in available_metrics
    ]

    # --------------------------------------------------------
    # Experiment + Run label
    # --------------------------------------------------------

    x_labels = [
        f"{row['experiment']}\n{row['run']}"
        for _, row in results.iterrows()
    ]

    z = results[available_metrics].T.values

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=metric_labels,
            text=z,
            texttemplate="%{text:.3f}",
            colorscale="Blues",
            zmin=score_range[0],
            zmax=score_range[1],
            showscale=True,
            colorbar=dict(
                title="Score",
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Experiment / Run",
        yaxis_title="Metric",
        width=1100,
        height=max(
            400,
            70 * len(available_metrics),
        ),
    )

    fig.show()


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(
    y_true,
    y_prob,
    title="ROC Curve",
    x_range=(0, 1),
    y_range=(0, 1),
):
    """
    Plot ROC curve for one model/run.

    x_range:
        False Positive Rate range.
        Default: (0, 1)

    y_range:
        True Positive Rate range.
        Default: (0, 1)
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
                dash="dash",
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(
            range=x_range,
        ),
        yaxis=dict(
            range=y_range,
        ),
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
    x_range=(0, 1),
    y_range=(0, 1),
):
    """
    Plot Precision-Recall curve for one model/run.

    x_range:
        Recall range.
        Default: (0, 1)

    y_range:
        Precision range.
        Default: (0, 1)
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
        xaxis=dict(
            range=x_range,
        ),
        yaxis=dict(
            range=y_range,
        ),
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
    n_images = len(sample_df)

    if n_images == 0:
        print("No error images found.")
        return

    # --------------------------------------------------------
    # Dynamic Grid
    # --------------------------------------------------------

    n_cols = min(4, n_images)
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    axes = axes.flatten()

    # Hide unused axes
    for ax in axes[n_images:]:
        ax.axis("off")

    # --------------------------------------------------------
    # Display Images
    # --------------------------------------------------------

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

    x_labels = [
        f"{row['experiment']}\n{row['run']}"
        for _, row in results.iterrows()
    ]

    fig = make_subplots(
        rows=1,
        cols=len(results),
        subplot_titles=x_labels,
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
        width=max(
            900,
            350 * len(results),
        ),
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
    x_range=(0, 1),
    y_range=(0, 1),
):
    """
    Compare ROC curves across experiment runs.

    prediction_results:
        {
            (experiment, run): {
                "labels": ...,
                "predictions": ...,
                "probabilities": ...,
            }
        }

    x_range:
        False Positive Rate range.
        Default: (0, 1)

    y_range:
        True Positive Rate range.
        Default: (0, 1)
    """

    fig = go.Figure()

    for key, result in prediction_results.items():

        if isinstance(key, tuple):
            experiment, run = key
            label = f"{experiment}\n{run}"
        else:
            # Backward compatibility for run-only dictionaries
            label = str(key)

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
                name=f"{label} (AUC={roc_auc:.4f})",
            )
        )

    # --------------------------------------------------------
    # Random Classifier
    # --------------------------------------------------------

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
        xaxis=dict(
            range=x_range,
        ),
        yaxis=dict(
            range=y_range,
        ),
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
    x_range=(0, 1),
    y_range=(0, 1),
):
    """
    Compare Precision-Recall curves across experiment runs.

    prediction_results:
        {
            (experiment, run): {
                "labels": ...,
                "predictions": ...,
                "probabilities": ...,
            }
        }

    x_range:
        Recall range.
        Default: (0, 1)

    y_range:
        Precision range.
        Default: (0, 1)
    """

    fig = go.Figure()

    for key, result in prediction_results.items():

        if isinstance(key, tuple):
            experiment, run = key
            label = f"{experiment}\n{run}"
        else:
            # Backward compatibility for run-only dictionaries
            label = str(key)

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
                name=f"{label} (PR-AUC={pr_auc:.4f})",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(
            range=x_range,
        ),
        yaxis=dict(
            range=y_range,
        ),
        width=850,
        height=600,
    )

    fig.show()