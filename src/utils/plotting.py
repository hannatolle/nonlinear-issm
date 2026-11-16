"""Plotting utilities for predictions, flow fields, training curves and model comparisons."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from src.utils.style import (
    TICK_LABEL_SIZE, MATH_LABEL_SIZE,
    LINEAR_COLOR, NONLINEAR_COLOR,
    STIM_ALPHA, STD_RANGE_FACTOR,
)
from src.utils.stats import significant_keys
import numpy as np
from typing import Optional, Any, Dict, Union, List, Tuple
import torch

SIG_ASTERISK_SIZE = 9


# Plotting helpers -------------------------------------------------------------

def get_cmap(n: int, name: str = "rainbow"):
    """Colormap by name; for n > 2, a discrete map of n evenly spaced colours."""
    try:
        base_cmap = plt.colormaps[name]
    except (AttributeError, KeyError):
        base_cmap = plt.cm.get_cmap(name)
    if n > 2:
        colors = base_cmap(np.linspace(0, 1, n))
        return ListedColormap(colors)
    return base_cmap

# Plotting functions -------------------------------------------------------------

def _render_stacked_panel(
    ax: plt.Axes,
    true_panel: np.ndarray,
    pred_panel: np.ndarray,
    u_active: np.ndarray,
    std_panel: Optional[np.ndarray] = None,
    panel_title: str = "",
    true_label: str = "Obs.",
    pred_label: str = "Pred.",
    show_legend: bool = True,
    fixed_channel_height: Optional[float] = None,
    inp_panel: Optional[np.ndarray] = None,
    mask_panel: Optional[np.ndarray] = None,
) -> None:
    """Render vertically stacked channels (true vs pred) into ax.

    Args:
        true_panel, pred_panel: (T, n_ch); std_panel optionally gives a +/- std band.
        u_active: (T,) bool, True at intervention timesteps.
        fixed_channel_height: fixed y-units per channel, so panels are comparable;
            None spaces channels by their own range.
        inp_panel: optional (T, n_ch) latent inputs (u @ B.T), drawn where non-zero.
        mask_panel: optional (T, n_ch) bool; masked cells of the true trace are drawn in gray.
    """
    T, n_ch = true_panel.shape

    margin = 1.0
    if fixed_channel_height is not None:
        shifts = np.arange(n_ch, dtype=float) * fixed_channel_height
    else:
        ch_ranges = np.nanmax(true_panel, axis=0) - np.nanmin(true_panel, axis=0)
        offsets = np.concatenate([[0.0], ch_ranges[:-1]])
        shifts = np.cumsum(offsets + margin)

    baselines = np.nanmin(true_panel, axis=0)  # (n_ch,)
    t_shifted = true_panel - baselines + shifts
    p_shifted = pred_panel - baselines + shifts

    for t in range(T):
        if u_active[t]:
            ax.axvline(x=t, color="gray", lw=0.8, alpha=STIM_ALPHA, zorder=0)

    for n in range(n_ch):
        if inp_panel is not None:
            inp_col = inp_panel[:, n]
            nz = inp_col != 0
            inp_shifted = inp_col - baselines[n] + shifts[n]
            inp_masked = np.where(nz, inp_shifted, np.nan)
            ax.plot(inp_masked, color="gold", lw=10.0, alpha=0.75, zorder=1,
                    label="Input" if (show_legend and n == 0) else "_nolegend_")
            # Single-sample inputs have no line segment, so mark them with a circle.
            isolated = nz & ~np.concatenate([[False], nz[:-1]]) & ~np.concatenate([nz[1:], [False]])
            if isolated.any():
                ts = np.flatnonzero(isolated)
                ax.plot(ts, inp_shifted[ts], linestyle="none", marker="o",
                        markersize=11, color="gold", alpha=0.5, zorder=1,
                        label="_nolegend_")
        ax.plot(t_shifted[:, n], color="k", lw=1.0, zorder=2,
                label=true_label if (show_legend and n == 0) else "_nolegend_")
        if mask_panel is not None:
            masked_true = np.where(mask_panel[:, n], t_shifted[:, n], np.nan)
            ax.plot(masked_true, color="gray", lw=1.5, zorder=3,
                    label="Masked" if (show_legend and n == 0) else "_nolegend_")
        if std_panel is not None:
            ax.fill_between(range(T),
                            p_shifted[:, n] - std_panel[:, n],
                            p_shifted[:, n] + std_panel[:, n],
                            color="red", alpha=0.2, zorder=3)
        ax.plot(p_shifted[:, n], color="red", lw=1.5, zorder=4,
                label=pred_label if (show_legend and n == 0) else "_nolegend_")

    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title(panel_title)
    if show_legend:
        ax.legend(loc="upper right", frameon=True)


def plot_y_predictions(u: torch.Tensor,
                       true: torch.Tensor,
                       pred: torch.Tensor,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       trial_to_plot: int = 0):
    """Predictions vs ground truth (B, T, N) for one trial, one subplot per channel."""
    B, T, M = u.shape
    N = true.shape[2]
    b = trial_to_plot

    fig, axs = plt.subplots(N, 1, figsize=(10, 3 * N))
    color_pred = "red"
    color_true = "k"
    if N == 1:
        axs = [axs]
    for n in range(N):
        for t in range(T):
            if torch.any(u[b, t] != 0):
                axs[n].axvspan(t-0.5, t+0.5, color='gray', alpha=STIM_ALPHA, zorder=0)
        axs[n].plot(true[b, :, n].detach().cpu().numpy(), color=color_true, label="true")
        axs[n].plot(pred[b, :, n].detach().cpu().numpy(), color=color_pred, label="pred")
        axs[n].set_title(f"Observation {n}")
        axs[n].set_xlabel("time")
        axs[n].set_ylabel("value")
        axs[n].legend()
    if title is not None:
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if save_path is not None:
        ext = save_path.split('.')[-1]
        plt.savefig(save_path, format=ext)
        plt.close('all')
    else:
        plt.show()


def plot_y_predictions_pair(
    u: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    idx1: int,
    idx2: int,
    trial_id1: Optional[int] = None,
    trial_id2: Optional[int] = None,
    n_dims_per_trial: int = 20,
    max_per_panel: int = 10,
    pred_label: str = "Pred.",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    mask1: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask2: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> None:
    """Predictions for two trials (batch indices idx1, idx2) side by side.

    Each trial shows its first n_dims_per_trial channels, max_per_panel per panel;
    trial_id1/2 label the panel titles. mask1/mask2 are optional (T, N) bool masks
    (True = ignore): fully masked channels are dropped, masked segments drawn in gray.
    """
    def _prep(idx, m):
        """(true, pred, mask) for one trial with fully masked channels dropped."""
        t = y_true[idx].detach().cpu().numpy()   # (T, N)
        p = y_pred[idx].detach().cpu().numpy()
        if m is not None:
            m = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
            keep_ch = ~m.all(axis=0)
            t, p, m = t[:, keep_ch], p[:, keep_ch], m[:, keep_ch]
        n_dims = min(n_dims_per_trial, t.shape[1])
        m = m[:, :n_dims] if m is not None else None
        return t[:, :n_dims], p[:, :n_dims], m

    true1, pred1, mask1 = _prep(idx1, mask1)
    true2, pred2, mask2 = _prep(idx2, mask2)
    u_active1 = torch.any(u[idx1] != 0, dim=-1).cpu().numpy()
    u_active2 = torch.any(u[idx2] != 0, dim=-1).cpu().numpy()

    n_panels_per_trial = int(np.ceil(max(true1.shape[1], true2.shape[1]) / max_per_panel))
    n_panels = 2 * n_panels_per_trial

    label1 = f"Trial {trial_id1}" if trial_id1 is not None else "Trial 1"
    label2 = f"Trial {trial_id2}" if trial_id2 is not None else "Trial 2"

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 1.2 * max_per_panel))
    if n_panels == 1:
        axes = [axes]

    for p in range(n_panels_per_trial):
        ch_start = p * max_per_panel
        for trial_offset, (true_t, pred_t, mask_t, u_act, label) in enumerate([
            (true1, pred1, mask1, u_active1, label1),
            (true2, pred2, mask2, u_active2, label2),
        ]):
            ch_end = min(ch_start + max_per_panel, true_t.shape[1])
            ax_idx = trial_offset * n_panels_per_trial + p
            panel_title = f"{label}, Ch {ch_start}–{ch_end - 1}"
            _render_stacked_panel(
                axes[ax_idx],
                true_t[:, ch_start:ch_end],
                pred_t[:, ch_start:ch_end],
                u_act,
                panel_title=panel_title,
                pred_label=pred_label,
                show_legend=(p == 0),
                mask_panel=(mask_t[:, ch_start:ch_end] if mask_t is not None else None),
            )

    plt.suptitle(title if title is not None else "")
    plt.tight_layout()
    if save_path is not None:
        ext = save_path.split(".")[-1]
        plt.savefig(save_path, format=ext)
        plt.close("all")
    else:
        plt.show()


def plot_multiple_y_predictions(
    u: torch.Tensor,
    true: torch.Tensor,
    preds: List[Tuple],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    trial_to_plot: int = 0,
    max_per_panel: Optional[int] = None,
    ax=None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    normalize_channels: bool = False,
    channel_margin: float = 1.0,
    obs_lw: float = 1.0,
    pred_lw: float = 1.5,
    mask_label: str = "Masked",
) -> None:
    """Stacked-channel plot of observations and several model predictions.

    Args:
        preds: list of (mean, std_or_None, color, label); arrays (B, T, N) or (T, N).
        max_per_panel: max channels per panel (None = one panel).
        mask: optional (B, T, N) or (T, N) bool (True = ignore); fully masked channels
            are dropped, masked segments drawn in gray and labelled mask_label.
        normalize_channels: scale each channel (truth and predictions alike) by its
            observed range, so channels of very different amplitude get equal slots.
        channel_margin: gap between channel slots, in units of channel height.
    """
    T = u.shape[1]
    N = true.shape[2]
    b = trial_to_plot

    def _np2d(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return x[b] if x.ndim == 3 else x  # (T, N)

    true_np = _np2d(true)
    preds_np = []
    for mean, std, color, label in preds:
        mean_np = _np2d(mean)
        std_np  = _np2d(std) if std is not None else None
        preds_np.append((mean_np, std_np, color, label))

    mask_np = None
    if mask is not None:
        mask_np = _np2d(mask)
        keep_ch = ~mask_np.all(axis=0)
        true_np = true_np[:, keep_ch]
        mask_np = mask_np[:, keep_ch]
        preds_np = [(mn[:, keep_ch], (sn[:, keep_ch] if sn is not None else None), c, l)
                    for (mn, sn, c, l) in preds_np]
    N = true_np.shape[1]

    if normalize_channels:
        rng = np.nanmax(true_np, axis=0) - np.nanmin(true_np, axis=0)
        rng = np.where(rng > 0, rng, 1.0)
        true_np = true_np / rng
        preds_np = [(mn / rng, (sn / rng if sn is not None else None), c, l)
                    for (mn, sn, c, l) in preds_np]

    u_active = torch.any(u[b] != 0, dim=-1).cpu().numpy()

    panel_size = max_per_panel if (max_per_panel is not None and N > max_per_panel) else N
    n_panels = int(np.ceil(N / panel_size))

    if ax is not None:
        axes = [ax]
        _own_fig = False
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 1.2 * panel_size))
        if n_panels == 1:
            axes = [axes]
        _own_fig = True

    for p in range(n_panels):
        ch_start = p * panel_size
        ch_end   = min(ch_start + panel_size, N)
        ax = axes[p]

        t_panel = true_np[:, ch_start:ch_end]
        n_ch = ch_end - ch_start

        margin = channel_margin
        ch_ranges = np.nanmax(t_panel, axis=0) - np.nanmin(t_panel, axis=0)
        offsets   = np.concatenate([[0.0], ch_ranges[:-1]])
        shifts    = np.cumsum(offsets + margin)
        t_shifted = t_panel - np.nanmin(t_panel, axis=0) + shifts

        for t in range(T):
            if u_active[t]:
                ax.axvline(x=t, color="gray", lw=0.8, alpha=STIM_ALPHA, zorder=0)

        m_panel = mask_np[:, ch_start:ch_end] if mask_np is not None else None
        for n in range(n_ch):
            ax.plot(t_shifted[:, n], color="k", lw=obs_lw,
                    label="Obs." if (p == 0 and n == 0) else "_nolegend_")
            if m_panel is not None:
                masked_true = np.where(m_panel[:, n], t_shifted[:, n], np.nan)
                ax.plot(masked_true, color="gray", lw=pred_lw, zorder=2,
                        label=mask_label if (p == 0 and n == 0) else "_nolegend_")

        for mean_np, std_np, color, label in preds_np:
            p_panel   = mean_np[:, ch_start:ch_end]
            p_shifted = p_panel - np.nanmin(t_panel, axis=0) + shifts
            for n in range(n_ch):
                first = p == 0 and n == 0
                if std_np is not None:
                    s_panel = std_np[:, ch_start:ch_end]
                    ax.fill_between(range(T),
                                    p_shifted[:, n] - s_panel[:, n],
                                    p_shifted[:, n] + s_panel[:, n],
                                    color=color, alpha=0.2, zorder=1)
                ax.plot(p_shifted[:, n], color=color, lw=pred_lw,
                        label=label if first else "_nolegend_")

        ax.set_yticks([])
        ax.set_xlabel("Time")
        ax.set_title(f"Ch {ch_start}–{ch_end - 1}")
        if p == 0:
            ax.legend(loc="upper right", frameon=True)

    if _own_fig:
        suptitle = title if title is not None else "y"
        plt.suptitle(suptitle)
        plt.tight_layout()
        if save_path is not None:
            ext = save_path.split(".")[-1]
            plt.savefig(save_path, format=ext)
            plt.close("all")
        else:
            plt.show()


def plot_x_predictions(x_true: torch.Tensor,
                       x_pred: torch.Tensor,
                       x_inputs: torch.Tensor,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None):
    """True vs predicted latents (B, T, D) of the first trial, on separate y-axes.

    x_inputs is dynamics.get_inputs(u), (B, T, D); non-zero marks driven latents.
    """
    D = x_true.shape[2]
    b = 0
    color_pred = "red"
    color_true = "k"

    fig, axs = plt.subplots(D, 1, figsize=(10, 3 * D))
    if D == 1:
        axs = [axs]

    for d in range(D):
        ax_pred = axs[d]
        ax_true = ax_pred.twinx()

        true_vals = x_true[b, :, d].detach().cpu().numpy()
        pred_vals = x_pred[b, :, d].detach().cpu().numpy()
        driven = x_inputs[b, :, d].detach().cpu().numpy() != 0

        T = true_vals.shape[0]
        for t in range(T):
            if driven[t]:
                ax_pred.axvspan(t - 0.5, t + 0.5, color="gray", alpha=STIM_ALPHA, zorder=0)

        ax_pred.plot(pred_vals, color=color_pred, lw=1.5, label="pred")
        ax_pred.set_ylabel(f"x{d + 1} pred", color=color_pred)
        ax_pred.tick_params(axis="y", labelcolor=color_pred)

        ax_true.plot(true_vals, color=color_true, lw=1.5, ls="-", label="true")
        ax_true.set_ylabel(f"x{d + 1} true", color=color_true)
        ax_true.tick_params(axis="y", labelcolor=color_true)

        ax_pred.set_xlabel("time")
        ax_pred.set_title(f"Latent {d + 1}")
        ax_pred.grid(True, alpha=0.3)

        lines_pred, labels_pred = ax_pred.get_legend_handles_labels()
        lines_true, labels_true = ax_true.get_legend_handles_labels()
        ax_pred.legend(lines_pred + lines_true, labels_pred + labels_true,
                       loc="upper right", fontsize=8)

    if title is not None:
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if save_path is not None:
        ext = save_path.split(".")[-1]
        plt.savefig(save_path, format=ext)
        plt.close("all")
    else:
        plt.show()


def plot_x_pred_only(x_pred: torch.Tensor,
                     x_inputs: torch.Tensor,
                     x_dyn: Optional[torch.Tensor] = None,
                     title: Optional[str] = None,
                     save_path: Optional[str] = None,
                     plot_together: bool = False,
                     max_per_panel: Optional[int] = None,
                     trial_idx: Optional[int] = None):
    """Encoder latents x_pred (B, T, D), optionally with dynamics latents x_dyn, no ground truth.

    Args:
        x_inputs: dynamics.get_inputs(u), (B, T, D); non-zero marks driven latents.
        plot_together: stack latents in panels with shared y-units (at most
            max_per_panel each), so low-variance latents show as flat; otherwise one
            subplot per latent.
        trial_idx: trial to plot; None picks the first stimulated trial (else 0).
    """
    D = x_pred.shape[2]

    if trial_idx is not None:
        b = trial_idx
    else:
        stim_trials = (x_inputs.abs().sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0]
        b = int(stim_trials[0]) if len(stim_trials) > 0 else 0

    if plot_together:
        enc_np   = x_pred[b].detach().cpu().numpy()                        # (T, D)
        dyn_np   = x_dyn[b].detach().cpu().numpy() if x_dyn is not None else enc_np.copy()
        inp_np   = x_inputs[b].detach().cpu().numpy()                      # (T, D)
        u_active = (inp_np != 0).any(axis=-1)                              # (T,)

        # Derive panel_size from n_panels so panels are equally filled.
        n_panels   = int(np.ceil(D / max_per_panel)) if (max_per_panel is not None and D > max_per_panel) else 1
        panel_size = int(np.ceil(D / n_panels))

        # One channel height for all panels so y-units are comparable.
        ch_ranges_all = np.nanmax(enc_np, axis=0) - np.nanmin(enc_np, axis=0)  # (D,)
        fixed_ch_height = float(np.nanmax(ch_ranges_all)) + 1.0

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 1.2 * panel_size), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for p in range(n_panels):
            ch_start = p * panel_size
            ch_end   = min(ch_start + panel_size, D)
            _render_stacked_panel(
                axes[p],
                enc_np[:, ch_start:ch_end],
                dyn_np[:, ch_start:ch_end],
                u_active,
                panel_title=f"Lat {ch_start}-{ch_end - 1}",
                true_label="Encoder",
                pred_label="Dynamics",
                show_legend=(p == 0),
                fixed_channel_height=fixed_ch_height,
                inp_panel=inp_np[:, ch_start:ch_end],
            )

        plt.suptitle(title if title is not None else "One-step latent predictions")
        plt.tight_layout()
    else:
        color_enc = "k"
        color_dyn = "red"

        fig, axs = plt.subplots(D, 1, figsize=(10, 3 * D))
        if D == 1:
            axs = [axs]

        for d in range(D):
            ax = axs[d]
            enc_vals = x_pred[b, :, d].detach().cpu().numpy()
            driven = x_inputs[b, :, d].detach().cpu().numpy() != 0

            T = enc_vals.shape[0]
            for t in range(T):
                if driven[t]:
                    ax.axvspan(t - 0.5, t + 0.5, color="gray", alpha=STIM_ALPHA, zorder=0)

            ax.plot(enc_vals, color=color_enc, lw=1.5, label="encoder")
            if x_dyn is not None:
                dyn_vals = x_dyn[b, :, d].detach().cpu().numpy()
                ax.plot(dyn_vals, color=color_dyn, lw=1.5, label="dynamics")
            ax.set_ylabel(f"x{d + 1}")
            ax.set_xlabel("time")
            ax.set_title(f"Latent {d + 1}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        if title is not None:
            plt.suptitle(title)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()

    if save_path is not None:
        ext = save_path.split(".")[-1]
        plt.savefig(save_path, format=ext)
        plt.close("all")
    else:
        plt.show()


def plot_flow_fields(funs: list[callable],
                     colors: list[Any] = ['k'],
                     xlim: tuple[float, float] = (-1, 1),
                     ylim: tuple[float, float] = (-1, 1),
                     n_points: int = 10,
                     ax: Optional[plt.Axes] = None,
                     max_arrow_length: Optional[float] = None,
                     alphas: Optional[list] = None,
                     widths: Optional[list] = None,
                     trajectory_starts: Optional[Union[torch.Tensor, np.ndarray]] = None,
                     trajectory_colors: Optional[list] = None,
                     trajectory_steps: int = 50,
                     trajectory_dt: float = 0.1,
                     trajectory_lw: float = 1.5,
                     trajectory_end_size: float = 7.0,
                     ) -> Tuple[plt.Figure, plt.Axes]:
    """Quiver plot of one or more 2D flow functions (N, 2) -> (N, 2); returns (fig, ax).

    Each field is scaled so its longest arrow has length max_arrow_length (default
    0.1 * the larger axis range). widths/alphas are per-field quiver settings.
    trajectory_starts, (2,) or (K, 2), adds one Euler-integrated trajectory per
    field and start (colours trajectory_colors, default colors), with a star at
    each endpoint unless trajectory_end_size <= 0.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    pts = np.stack((X.flatten(), Y.flatten())).T  # (n_grid, 2)

    target_length = max_arrow_length if max_arrow_length is not None else (
        0.1 * max(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    for i in range(len(funs)):
        fun = funs[i]
        color = colors[i]
        alpha = alphas[i] if alphas is not None else 1.0
        matrix = fun(pts).T  # (2, n_grid)
        norms = np.linalg.norm(matrix.T, axis=1)
        scale_f = target_length / (norms.max() + 1e-12)
        matrix = matrix * scale_f

        quiver_kw = {} if widths is None else {"width": widths[i]}
        Q = ax.quiver(
            X, Y,
            np.zeros_like(X), np.zeros_like(Y),
            pivot="mid",
            scale=1,
            color=color,
            alpha=alpha,
            **quiver_kw,
        )
        Q.set_UVC(matrix[0], matrix[1])

    if trajectory_starts is not None:
        traj_colors = trajectory_colors if trajectory_colors is not None else colors
        starts = (trajectory_starts.cpu().numpy() if torch.is_tensor(trajectory_starts)
                  else np.asarray(trajectory_starts, dtype=float))
        starts = np.atleast_2d(starts)  # (K, 2)
        for i in range(len(funs)):
            fun = funs[i]
            color = traj_colors[i]
            for x0 in starts:
                traj = np.empty((trajectory_steps + 1, 2))
                traj[0] = x0
                for t in range(trajectory_steps):
                    flow = np.asarray(fun(traj[t][None, :])).reshape(2)
                    traj[t + 1] = traj[t] + trajectory_dt * flow
                ax.plot(traj[:, 0], traj[:, 1], color=color, lw=trajectory_lw, zorder=3)
                if trajectory_end_size > 0:
                    ax.plot(traj[-1, 0], traj[-1, 1], marker="*", color=color,
                            markersize=trajectory_end_size, linestyle="None", zorder=5)
        ax.scatter(starts[:, 0], starts[:, 1], color=traj_colors[0], s=5, zorder=4)
        # Pin limits (with the default 5% margin) so trajectories leaving the window
        # clip instead of rescaling the equal-aspect axes.
        mx, my = 0.05 * (xlim[1] - xlim[0]), 0.05 * (ylim[1] - ylim[0])
        ax.set_xlim(xlim[0] - mx, xlim[1] + mx)
        ax.set_ylim(ylim[0] - my, ylim[1] + my)

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel("$x_0$", fontsize=MATH_LABEL_SIZE)
    ax.set_ylabel("$x_1$", fontsize=MATH_LABEL_SIZE)
    ax.set_aspect("equal")

    return fig, ax

def plot_training_metrics(history_df, save_path: Optional[str] = None, best_epoch: Optional[int] = None):
    """Train/val curves from a history DataFrame (epoch, train_*/val_* columns); returns the figure.

    Row 1: loss, log-likelihood, y MAE (rollout solid, encoder dashed).
    Row 2: log q, prior log p(x|u), KL = log q - log p(x|u).
    """
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()

    def _plot_curve(ax, key, ylabel, title, log_scale=False):
        train_col, val_col = f"train_{key}", f"val_{key}"
        if val_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[val_col], color="red", label="val")
        if train_col in history_df.columns:
            tr = history_df.dropna(subset=[train_col])
            if not tr.empty:
                ax.plot(tr["epoch"], tr[train_col], color="blue", label="train")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", ls="--", lw=1)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    _plot_curve(axs[0], "loss", "loss", "loss (negative ELBO)", log_scale=False)
    _plot_curve(axs[1], "log_likelihood", "log p(y|emission(x))", "log_likelihood")
    ax2 = axs[2]
    has_rollout = "val_y_mae_rollout" in history_df.columns
    has_train_rollout = (has_rollout
                         and "train_y_mae_rollout" in history_df.columns
                         and history_df["train_y_mae_rollout"].notna().any())
    has_enc = "val_y_mae" in history_df.columns
    has_train_enc = (has_enc
                     and "train_y_mae" in history_df.columns
                     and history_df["train_y_mae"].notna().any())
    if has_rollout:
        ax2.plot(history_df["epoch"], history_df["val_y_mae_rollout"], color="red")
    if has_train_rollout:
        tr_ro = history_df.dropna(subset=["train_y_mae_rollout"])
        ax2.plot(tr_ro["epoch"], tr_ro["train_y_mae_rollout"], color="blue")
    if has_enc:
        ax2.plot(history_df["epoch"], history_df["val_y_mae"], color="red", ls="--")
    if has_train_enc:
        tr_enc = history_df.dropna(subset=["train_y_mae"])
        ax2.plot(tr_enc["epoch"], tr_enc["train_y_mae"], color="blue", ls="--")
    if best_epoch is not None:
        ax2.axvline(best_epoch, color="gray", ls="--", lw=1)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("y MAE")
    ax2.set_title("y MAE")
    ax2.grid(True)
    if has_enc:
        ax2.legend(handles=[
            Line2D([0], [0], color="black", lw=2, label="rollout"),
            Line2D([0], [0], color="black", lw=2, ls="--", label="encoder"),
        ], fontsize=8, loc="upper right")

    _plot_curve(axs[3], "log_q", "log q(x|y,u)", "log_q_total")
    _plot_curve(axs[4], "prior_log_prob", "log p(x|u)", "prior_log_prob")
    if "val_log_q" in history_df.columns and "val_prior_log_prob" in history_df.columns:
        val_kl = history_df["val_log_q"] - history_df["val_prior_log_prob"]
        axs[5].plot(history_df["epoch"], val_kl, color="red", label="val")
    if "train_log_q" in history_df.columns and "train_prior_log_prob" in history_df.columns:
        tr_kl = history_df.dropna(subset=["train_log_q", "train_prior_log_prob"])
        if not tr_kl.empty:
            train_kl = tr_kl["train_log_q"] - tr_kl["train_prior_log_prob"]
            axs[5].plot(tr_kl["epoch"], train_kl, color="blue", label="train")
    if best_epoch is not None:
        axs[5].axvline(best_epoch, color="gray", ls="--", lw=1)
    axs[5].set_xlabel("epoch")
    axs[5].set_ylabel("KL")
    axs[5].set_title("KL divergence (log_q - prior_log_prob)")
    axs[5].grid(True)
    if "beta" in history_df.columns:
        ax_beta = axs[5].twinx()
        ax_beta.plot(history_df["epoch"], history_df["beta"], color="green", lw=1.5, ls="--", label="beta")
        ax_beta.set_ylabel("beta", color="green")
        ax_beta.tick_params(axis="y", labelcolor="green")

    legend_handles = [
        Line2D([0], [0], color="blue", lw=2, label="train"),
        Line2D([0], [0], color="red", lw=2, label="val"),
    ]
    if best_epoch is not None:
        legend_handles.append(Line2D([0], [0], color="gray", lw=1, ls="--", label="best weights"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        ext = save_path.split(".")[-1]
        plt.savefig(save_path, format=ext)
        plt.close("all")
    else:
        plt.show()

    return fig
        
def std_clipped_ylim(ref_lo: float, ref_hi: float, band_lo: float, band_hi: float,
                     factor: float = STD_RANGE_FACTOR,
                     margin: float = 0.05) -> Optional[Tuple[float, float]]:
    """y-limits capped at factor x the lines' own range (ref_lo, ref_hi); wider std bands clip.

    Stops a blown-up posterior std from flattening every line. The extra headroom is
    split in proportion to the band's overshoot on each side; margin pads the result.
    """
    ref = ref_hi - ref_lo
    if not np.isfinite(ref) or ref <= 0:
        return None
    below = max(0.0, ref_lo - band_lo)
    above = max(0.0, band_hi - ref_hi)
    budget = (factor - 1.0) * ref
    if below + above > budget:
        scale = budget / (below + above)
        below, above = below * scale, above * scale
    lo, hi = ref_lo - below, ref_hi + above
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


def plot_dual_mean_predictions(
    u: Union[np.ndarray, "torch.Tensor"],
    y_true: Union[np.ndarray, "torch.Tensor"],
    pred_mean_1: Union[np.ndarray, "torch.Tensor"],
    pred_std_1: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    color_1: str = "C0",
    pred_mean_2: Union[np.ndarray, "torch.Tensor"] = None,
    pred_std_2: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    color_2: str = "C1",
    axs: Optional[List[plt.Axes]] = None,
    variable_name: str = "var",
    label_1: str = "model 1",
    label_2: str = "model 2",
    show_legend: bool = False,
    t_axis: Optional[np.ndarray] = None,
    std_range_factor: Optional[float] = None,
    lw_true: float = 2.0,
    lw_pred: float = 1.5,
) -> List[plt.Axes]:
    """Ground truth and two models' mean +/- std, one axis per channel; returns the N axes.

    u is (T, M), y_true and predictions (T, N); gray spans mark non-zero u.
    std_range_factor, if set, caps each y-range via std_clipped_ylim.
    """
    def _np(a):
        return a.detach().cpu().numpy() if hasattr(a, "detach") and callable(a.detach) else np.asarray(a)

    u = _np(u)
    y_true = _np(y_true)
    pred_mean_1 = _np(pred_mean_1)
    pred_mean_2 = _np(pred_mean_2) if pred_mean_2 is not None else None
    pred_std_1 = _np(pred_std_1) if pred_std_1 is not None else None
    pred_std_2 = _np(pred_std_2) if pred_std_2 is not None else None

    T, n_series = y_true.shape

    if t_axis is None:
        t_axis = np.arange(T)
    step = float(t_axis[1] - t_axis[0]) if T > 1 else 1.0
    xlabel = "Time (s)" if step != 1.0 else "Time step"

    if axs is None:
        _, _axs = plt.subplots(n_series, 1, figsize=(10, 3 * n_series), squeeze=False)
        axs = list(_axs.flatten())

    for idx in range(n_series):
        ax = axs[idx]
        for t in range(T):
            if np.any(u[t] != 0):
                ax.axvspan(t_axis[t] - step / 2, t_axis[t] + step / 2,
                           color="lightgray", alpha=STIM_ALPHA, zorder=0)
        ax.plot(t_axis, y_true[:, idx], color="black", lw=lw_true, label="true", zorder=1)

        lines = [y_true[:, idx], pred_mean_1[:, idx]]
        if pred_mean_2 is not None:
            lines.append(pred_mean_2[:, idx])
        ref_lo = min(float(np.nanmin(l)) for l in lines)
        ref_hi = max(float(np.nanmax(l)) for l in lines)
        band_lo, band_hi = ref_lo, ref_hi

        for mean, std, color, label in ((pred_mean_1, pred_std_1, color_1, label_1),
                                        (pred_mean_2, pred_std_2, color_2, label_2)):
            if mean is None:
                continue
            if std is not None:
                ax.fill_between(t_axis,
                                mean[:, idx] - std[:, idx],
                                mean[:, idx] + std[:, idx],
                                color=color, alpha=0.25, zorder=2)
                band_lo = min(band_lo, float(np.nanmin(mean[:, idx] - std[:, idx])))
                band_hi = max(band_hi, float(np.nanmax(mean[:, idx] + std[:, idx])))
            ax.plot(t_axis, mean[:, idx], color=color, lw=lw_pred, label=label, zorder=3)

        if std_range_factor is not None:
            ylim = std_clipped_ylim(ref_lo, ref_hi, band_lo, band_hi, std_range_factor)
            if ylim is not None:
                ax.set_ylim(*ylim)

        ax.set_ylabel(f"${variable_name}_{{{idx}}}$", fontsize=MATH_LABEL_SIZE)
        if idx == len(axs) - 1:
            ax.set_xlabel(xlabel)
        if show_legend:
            ax.legend(loc="upper right", fontsize=8)

    return axs


OUTLIER_PIN_FRAC = 0.04   # outlier pin inset from the edge, as a fraction of the range


def coarsest_ticks(lo: float, hi: float, min_ticks: int = 3,
                   max_decimals: int = 2) -> Tuple[np.ndarray, int]:
    """(ticks, decimals) of the coarsest round grid with at least min_ticks ticks in [lo, hi].

    Falls back to the finest grid tried.
    """
    lo, hi = (hi, lo) if lo > hi else (lo, hi)
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.array([lo]), max(max_decimals, 0)
    eps = 1e-9 * max(1.0, abs(hi), abs(lo))
    # Start coarser than the range, so e.g. 3..52 can be labelled in tens.
    finest = (np.array([lo, hi]), max_decimals)
    for d in range(-int(np.ceil(np.log10(span))), max_decimals + 1):
        for step in (5 * 10.0 ** -d, 2 * 10.0 ** -d, 1 * 10.0 ** -d):
            if step > span:
                continue
            first = np.ceil(lo / step - eps) * step
            ticks = np.round(np.arange(first, hi + eps, step), max(d, 0) + 1) + 0.0
            ticks = ticks[(ticks >= lo - eps) & (ticks <= hi + eps)]
            finest = (ticks, max(d, 0))
            if len(ticks) >= min_ticks:
                return np.round(ticks, max(d, 0)) + 0.0, max(d, 0)
    return np.round(finest[0], finest[1]) + 0.0, finest[1]


def coarsest_axis_ticks(ax: plt.Axes, axis: str = "y", min_ticks: int = 3,
                        max_decimals: int = 2) -> None:
    """Apply coarsest_ticks to an axis, keeping its limits."""
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    ticks, decimals = coarsest_ticks(lo, hi, min_ticks, max_decimals)
    labels = [f"{t:.{decimals}f}" for t in ticks]
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_ylim(lo, hi)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(lo, hi)


def coarse_axis_ticks(ax: plt.Axes, axis: str = "y", decimals: int = 2,
                      max_ticks: int = 6, min_ticks: int = 3,
                      fontsize: Optional[float] = None) -> None:
    """Snap ticks to a grid no finer than 10^-decimals, labelled to that precision.

    Takes the finest multiple of 10^-decimals giving at most max_ticks ticks. The axis
    is left alone if it is already on the grid or fewer than min_ticks would remain.
    """
    q = 10.0 ** -decimals
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    current = [t for t in (ax.get_yticks() if axis == "y" else ax.get_xticks())
               if lo <= t <= hi]
    if current and all(abs(t / q - round(t / q)) < 1e-6 for t in current):
        return

    for step in (q, 2 * q, 5 * q, 10 * q, 20 * q, 25 * q, 50 * q, 100 * q):
        ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        if len(ticks) <= max_ticks:
            break
    if len(ticks) < min_ticks:
        return
    ticks[np.isclose(ticks, 0.0)] = 0.0     # else np.ceil gives -0.0, labelled "-0.00"

    labels = [f"{t:.{decimals}f}" for t in ticks]
    kw = {} if fontsize is None else {"fontsize": fontsize}
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, **kw)
        ax.set_ylim(lo, hi)          # set_yticks can nudge the limits; restore them
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, **kw)
        ax.set_xlim(lo, hi)


def robust_panel_range(vals: np.ndarray, factor: float = 1.5,
                       pad: float = 0.05,
                       gutter: float = 0.18) -> Tuple[float, float, list]:
    """(ylo, yhi, outliers) for one panel, excluding lone axis-dominating extremes.

    An extreme is an outlier if its gap to the rest exceeds factor x the span of the
    others. Limits get pad headroom, plus a gutter on each outlier side for pin_outliers.
    """
    v = np.sort(vals[np.isfinite(vals)])
    if v.size == 0:
        return 0.0, 1.0, []
    outliers = []
    lo_i, hi_i = 0, v.size - 1
    if v.size >= 4:
        span_lo = v[-1] - v[1]
        if span_lo > 0 and (v[1] - v[0]) > factor * span_lo:
            outliers.append(float(v[0])); lo_i = 1
        span_hi = v[-2] - v[0]
        if span_hi > 0 and (v[-1] - v[-2]) > factor * span_hi:
            outliers.append(float(v[-1])); hi_i = v.size - 2
    inl = v[lo_i:hi_i + 1]
    span = (inl[-1] - inl[0]) or 1.0
    ylo = inl[0]  - pad * span - (gutter * span if lo_i > 0 else 0.0)
    yhi = inl[-1] + pad * span + (gutter * span if hi_i < v.size - 1 else 0.0)
    return ylo, yhi, outliers


def pin_outliers(ax: plt.Axes, x: float, values, ylo: float, yhi: float,
                 color, alpha: float = 1.0, side: int = 1,
                 fontsize: float = TICK_LABEL_SIZE - 2) -> None:
    """Draw values outside [ylo, yhi] as labelled 'X' markers pinned near the edge.

    side (+1/-1) sets the label offset direction.
    """
    span = yhi - ylo
    for o in np.atleast_1d(values):
        o = float(o)
        if ylo <= o <= yhi:
            continue
        y = (ylo + OUTLIER_PIN_FRAC * span if o < ylo
             else yhi - OUTLIER_PIN_FRAC * span)
        ax.scatter([x], [y], marker="X", s=34, color=color, alpha=alpha,
                   edgecolor="black", linewidth=0.5, zorder=5)
        ax.annotate(f"{o:.2f}", (x, y), textcoords="offset points",
                    xytext=(5 * side, 0), ha="left" if side > 0 else "right",
                    va="center", fontsize=fontsize, color="black")


def draw_matrix_heatmap(
    ax: plt.Axes,
    mat: Optional[np.ndarray],
    title: str,
    std_mat: Optional[np.ndarray] = None,
    cmap: str = "YlGnBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    n_decimals: int = 2,
) -> None:
    """Draw an annotated heatmap with cell values (and optional ±std) on ax."""
    if mat is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    vmax_use = vmax if vmax is not None else float(np.nanmax(np.abs(mat)))
    vmin_use = vmin if vmin is not None else 0.0

    im = ax.imshow(mat, aspect="equal", cmap=cmap, vmin=vmin_use, vmax=vmax_use)
    ax.set_title(title, fontsize=9)
    nrows, ncols = mat.shape
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels([str(j) for j in range(ncols)], fontsize=6)
    ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=6)

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fs = max(4, min(8, 60 // max(nrows, ncols)))
    for i in range(nrows):
        for j in range(ncols):
            val  = mat[i, j]
            rgba = im.cmap(im.norm(val))
            lum  = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            tc   = "black" if lum > 0.5 else "white"
            if std_mat is not None:
                label = f"{val:.{n_decimals}f}\n±{std_mat[i, j]:.{n_decimals}f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=max(3, fs - 1), color=tc)
            else:
                ax.text(j, i, f"{val:.{n_decimals}f}", ha="center", va="center",
                        fontsize=fs, color=tc)


# ---------------------------------------------------------------------------
# Paired-tier comparison (two-model, per-trial metrics)
# ---------------------------------------------------------------------------

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        adj[i] = min(running_min, float(pvals[i]) * n / (rank + 1))
        running_min = adj[i]
    return np.clip(adj, 0.0, 1.0)


def significance_stars(p: float, ns: str = "ns") -> str:
    """Star string for a p-value; ns for non-significant or missing."""
    if p is None or not np.isfinite(p):
        return ns
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ns


def plot_paired_tier_comparison(
    df1: "pd.DataFrame",
    df2: "pd.DataFrame",
    name1: str = "Model 1",
    name2: str = "Model 2",
    color1: str = LINEAR_COLOR,
    color2: str = NONLINEAR_COLOR,
    metrics: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    axes: Optional[List[plt.Axes]] = None,
    save_path: Optional[str] = None,
    show_titles: bool = True,
    records: Optional[List[dict]] = None,
    tier_labels: Optional[Dict[str, str]] = None,
    point_size: float = 20,
    median_lw: float = 2,
    star_size: float = 12,
    tick_label_size: float = 9,
    legend_size: float = 8,
    title_size: float = 8,
    line_width: float = 0.5,
) -> "plt.Figure":
    """Paired per-trial boxplots of two models, one subplot per metric, tiers on x.

    df1/df2 are multi_tier_eval_metrics.csv tables, paired on trial_idx. Significance
    is only drawn from precomputed records (dicts keyed metric, x_value=tier,
    test_statistic, cohen_d, p_raw, significant, n_pairs); no test is run here.
    """
    if metrics is None:
        metrics = ["y_corr", "y_corr_rollout", "y_mae", "y_mae_rollout"]
    if metric_labels is None:
        metric_labels = {
            "y_corr":         "Reconstruction accuracy",
            "y_corr_rollout": "Rollout accuracy",
            "y_mae":          "Reconstruction MAE",
            "y_mae_rollout":  "Rollout MAE",
        }

    merged = df1.merge(df2, on=["trial_idx", "trial_tier"], suffixes=("_1", "_2"))

    sig_by = {(r.get("metric"), r.get("x_value")): r for r in (records or [])}

    _own_fig = axes is None
    if _own_fig:
        fig, ax_arr = plt.subplots(2, 2, figsize=(13, 10))
        axes_flat = list(ax_arr.flatten())
    else:
        axes_flat = list(axes)
        fig = axes_flat[0].figure

    rng = np.random.default_rng(0)
    tiers = sorted(merged["trial_tier"].unique())

    for ax, metric in zip(axes_flat, metrics):
        col1 = f"{metric}_1"
        col2 = f"{metric}_2"
        if col1 not in merged.columns or col2 not in merged.columns:
            ax.set_visible(False)
            continue

        tier_recs = [sig_by.get((metric, tier)) for tier in tiers]

        x_centers = np.arange(len(tiers), dtype=float) * 3.0
        offset = 0.4
        bp_kw = dict(widths=0.6, patch_artist=True, showfliers=False,
                     medianprops=dict(color="white", lw=median_lw))

        y_tops = []

        for ti, (tier, x_c) in enumerate(zip(tiers, x_centers)):
            mask = merged["trial_tier"] == tier
            sub = merged.loc[mask, [col1, col2]].dropna()
            v1, v2 = sub[col1].values, sub[col2].values
            if len(v1) == 0:
                y_tops.append(ax.get_ylim()[1] if ax.get_ylim() != (0, 1) else 0)
                continue

            x1, x2 = x_c - offset, x_c + offset

            for a, b in zip(v1, v2):
                ax.plot([x1, x2], [a, b],
                        color="lightgray", lw=line_width, zorder=1, alpha=0.7)

            for xp, vals, color in [(x1, v1, color1), (x2, v2, color2)]:
                bp = ax.boxplot([vals], positions=[xp], **bp_kw)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                    patch.set_edgecolor(color)

            jitter1 = rng.normal(0, 0.08, len(v1))
            jitter2 = rng.normal(0, 0.08, len(v2))
            ax.scatter(x1 + jitter1, v1, color=color1, s=point_size, alpha=0.8,
                       edgecolors="none", zorder=4)
            ax.scatter(x2 + jitter2, v2, color=color2, s=point_size, alpha=0.8,
                       edgecolors="none", zorder=4)

            y_tops.append(float(np.nanmax(np.concatenate([v1, v2]))))

        ax.set_xlim(x_centers[0] - 1.2, x_centers[-1] + 1.2)
        y0, y1_ax = ax.get_ylim()
        span = (y1_ax - y0) if y1_ax != y0 else 1.0
        for ti, x_c in enumerate(x_centers):
            rec = tier_recs[ti]
            if rec is not None and rec.get("significant"):
                y_ann = y_tops[ti] + 0.04 * span
                ax.text(x_c, y_ann, "*",
                        ha="center", va="bottom", color="black",
                        fontsize=star_size, zorder=5)

        ax.set_xticks(x_centers)
        ax.set_xticklabels([(tier_labels or {}).get(t, t) for t in tiers],
                           fontsize=tick_label_size)

        if metric == metrics[0]:
            import matplotlib.patches as mpatches
            ax.legend(
                handles=[
                    mpatches.Patch(color=color1, label=name1, alpha=0.7),
                    mpatches.Patch(color=color2, label=name2, alpha=0.7),
                ],
                fontsize=legend_size, framealpha=0.7, loc="best",
            )

        if show_titles and records is not None:
            label = metric_labels.get(metric, metric)
            stats_lines = []
            for tier, rec in zip(tiers, tier_recs):
                if rec is None:
                    stats_lines.append(f"{tier}: n/a")
                    continue
                w_s, d_s = rec.get("test_statistic"), rec.get("cohen_d")
                p_s, n_s = rec.get("p_raw"), rec.get("n_pairs")
                sig_mark = "*" if rec.get("significant") else ""
                p_str = f"{p_s:.3f}" if p_s is not None and np.isfinite(p_s) else "nan"
                d_str = f"{d_s:.2f}" if d_s is not None and np.isfinite(d_s) else "nan"
                w_str = f"{w_s:.2f}" if w_s is not None and np.isfinite(w_s) else "nan"
                stats_lines.append(
                    f"{tier}: W={w_str}, d={d_str}, p={p_str}{sig_mark}, n={n_s}"
                )
            title_str = label + "\n" + "\n".join(stats_lines)
            ax.set_title(title_str, fontsize=title_size, loc="left")
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if _own_fig:
        fig.suptitle(f"{name1} vs {name2}", fontsize=11, y=1.01)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

    return fig
    return save_path


# ---------------------------------------------------------------------------
# Sweep line plots (metric vs a swept config parameter)
# ---------------------------------------------------------------------------

def _compute_pvalues(records1: list, records2: list) -> dict:
    """Uncorrected paired Wilcoxon p per shared x of two load_sweep record lists (paired on seed)."""
    from src.utils.stats import paired_wilcoxon
    map1 = {r[0]: r[3] for r in records1}
    map2 = {r[0]: r[3] for r in records2}
    pvals = {}
    for x in sorted(set(map1) & set(map2)):
        a, b = map1[x], map2[x]
        if len(a) < 3 or len(b) < 3 or len(a) != len(b):
            continue
        res = paired_wilcoxon(a, b)
        if res is not None:
            pvals[x] = res[1]
    return pvals


def plot_sweep(metric_entries, config_param, name1, name2,
               color1, color2, output_dir, fmt, ax=None,
               xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               metric_labels: Optional[Dict[str, str]] = None,
               legend: bool = True, significance: str = "markers",
               records: Optional[list] = None):
    """Metrics vs a swept config parameter for two models.

    metric_entries are (records1, records2, metric, linestyle) with load_sweep records.
    records: precomputed, corrected test records (see stats.significant_keys); if None,
    paired Wilcoxon tests are run here and BH-FDR corrected over all metrics and x.
    significance="markers" draws open markers for non-significant points; "asterisks"
    puts a '*' above each significant x. xlabel/ylabel/metric_labels only change
    labels; the output filename is always derived from config_param.
    """
    if significance not in ("markers", "asterisks"):
        raise ValueError(f"Unknown significance style: {significance!r}")
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # sig_set holds significant (entry_idx, x) pairs, from injected records or tests run here.
    injected = significant_keys(records)
    sig_set: set = set()
    if injected is None:
        all_keys, all_pvals = [], []
        for entry_idx, (records1, records2, _, _) in enumerate(metric_entries):
            for x, p in _compute_pvalues(records1, records2).items():
                all_keys.append((entry_idx, x))
                all_pvals.append(p)
        has_tests = bool(all_pvals)
        if has_tests:
            corrected = _bh_fdr(np.array(all_pvals))
            sig_set = {k for k, cp in zip(all_keys, corrected) if cp < 0.05}
    else:
        has_tests = True
        sig_set = {
            (entry_idx, x)
            for entry_idx, (records1, records2, metric, _) in enumerate(metric_entries)
            for x in ({r[0] for r in records1} & {r[0] for r in records2})
            if (metric, round(float(x), 9)) in injected
        }

    for entry_idx, (records1, records2, _, linestyle) in enumerate(metric_entries):
        for records, name, color in [(records1, name1, color1), (records2, name2, color2)]:
            if not records:
                continue
            xs   = [r[0] for r in records]
            ys   = [r[1] for r in records]
            errs = [r[2] for r in records]
            label = name if len(metric_entries) == 1 else None
            ax.errorbar(xs, ys, yerr=errs, label=label, color=color,
                        linestyle=linestyle, marker=None, linewidth=1.5,
                        capsize=3, capthick=1.2, elinewidth=1.0)
            # Filled if significant or no tests ran; always filled in "asterisks" mode.
            for x, y in zip(xs, ys):
                is_sig = (significance == "asterisks") or (not has_tests) \
                    or ((entry_idx, x) in sig_set)
                ax.plot(x, y, marker="o", markersize=5, linestyle="none",
                        color=color,
                        markerfacecolor=color if is_sig else "white",
                        markeredgecolor=color, markeredgewidth=1.2, zorder=3)

    if significance == "asterisks" and sig_set:
        # One asterisk height per panel, above all error bars.
        tops = [r[1] + r[2] for r1, r2, _, _ in metric_entries
                for records in (r1, r2) for r in records]
        ylo, yhi = ax.get_ylim()
        span = (yhi - ylo) or 1.0
        y_ast = max(tops) + 0.04 * span
        for x in sorted({x for _, x in sig_set}):
            ax.text(x, y_ast, "*", ha="center", va="bottom",
                    fontsize=SIG_ASTERISK_SIZE, color="black", zorder=5)
        ax.set_ylim(ylo, max(yhi, y_ast + 0.14 * span))

    labels = metric_labels or {}
    def _mlabel(metric: str) -> str:
        return labels.get(metric, metric)

    param_label = config_param.split(".")[-1]
    ax.set_xlabel(xlabel if xlabel is not None else param_label)
    metrics_label = ", ".join(_mlabel(e[2]) for e in metric_entries)
    ax.set_ylabel(ylabel if ylabel is not None else metrics_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend and len(metric_entries) > 1:
        color_handles = [
            mpatches.Patch(color=color1, label=name1),
            mpatches.Patch(color=color2, label=name2),
        ]
        style_handles = [
            mlines.Line2D([], [], color="black", linestyle=e[3], label=_mlabel(e[2]))
            for e in metric_entries
        ]
        ax.legend(handles=color_handles + style_handles, frameon=False)
    elif legend:
        ax.legend(frameon=False)

    if _own_fig:
        import os
        from pathlib import Path
        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        metrics_slug = "_".join(e[2] for e in metric_entries)
        out_path = Path(output_dir) / f"{param_label}_vs_{metrics_slug}.{fmt}"
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")
        plt.close(fig)
