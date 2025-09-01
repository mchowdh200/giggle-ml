import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def _get_text_color_for_background(bg_color_rgb):
    luminance = 0.299 * bg_color_rgb[0] + 0.587 * bg_color_rgb[1] + 0.114 * bg_color_rgb[2]
    return "white" if luminance < 0.5 else "black"


def plot_heatmap_with_averages(
    x_labels,
    y_label_groups,
    data_matrix,
    figsize=None,  # Ignored if desired_tile_in is set and creating a new figure
    desired_tile_in=None,  # Tuple (width_inches, height_inches) for heatmap cells
    cmap_name="viridis",
    title="Heatmap with Averages",
    fig_creation_margins=None,
    fig=None,
    subplot_spec=None,
):
    """
    Generates a heatmap with configurable tile sizes, row/column averages, and grouped Y-labels.
    - Y-averages on left; X-averages on bottom. Labels are on these strips.
    - Average strips are scaled proportionally to one heatmap cell.
    - Dotted lines separate Y-axis groups.
    - Space between heatmap and average strips is minimal.

    Args:
        x_labels (list of str): Labels for the x-axis.
        y_label_groups (list of tuple[str, int]): Y-axis group labels. (group_name, num_rows).
        data_matrix (np.ndarray or list of lists): 2D matrix of float values.
        figsize (tuple, optional): Figure size if creating a new figure AND desired_tile_in is None.
                                   Ignored if desired_tile_in is set for a new figure.
        desired_tile_in (tuple, optional): (width_inches, height_inches) for each heatmap cell.
                                          If set for a new figure, figsize is calculated.
                                          If set for an existing figure, ratios are proportional.
        cmap_name (str, optional): Colormap name.
        title (str, optional): Title for this heatmap instance.
        fig_creation_margins (dict, optional): Margins for new figure (e.g., {'left':0.1}).
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        subplot_spec (matplotlib.gridspec.SubplotSpec, optional): SubplotSpec to draw into.
                                                                  Required if 'fig' is provided.
    Returns:
        matplotlib.figure.Figure: The (potentially new) Figure object.
        tuple: Axes (ax_heatmap, ax_row_avg, ax_col_avg, ax_grand_avg, ax_cbar).
    """
    data = np.asarray(data_matrix, dtype=float)

    num_rows_from_groups = sum(count for _, count in y_label_groups) if y_label_groups else 0
    if num_rows_from_groups != data.shape[0]:
        raise ValueError(
            f"Sum of counts in y_label_groups ({num_rows_from_groups}) "
            f"does not match data_matrix rows ({data.shape[0]})."
        )

    num_rows, num_cols = data.shape

    if len(x_labels) != num_cols:
        raise ValueError(f"x_labels length ({len(x_labels)}) != data columns ({num_cols}).")

    # --- Calculate Averages & Normalization (as before) ---
    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(data, axis=1, keepdims=True)
        col_means = np.nanmean(data, axis=0, keepdims=True)
        grand_mean_val = np.nanmean(data)
    # ... (vmin_calc, vmax_calc, norm, cmap_obj setup as before) ...
    all_finite_values_list = []
    if not np.all(np.isnan(data)):
        all_finite_values_list.append(data[~np.isnan(data)].flatten())
    if not np.all(np.isnan(row_means)):
        all_finite_values_list.append(row_means[~np.isnan(row_means)].flatten())
    if not np.all(np.isnan(col_means)):
        all_finite_values_list.append(col_means[~np.isnan(col_means)].flatten())
    if not np.isnan(grand_mean_val):
        all_finite_values_list.append(np.array([grand_mean_val]))

    vmin_calc, vmax_calc = 0, 1
    if all_finite_values_list:
        all_finite_values = np.concatenate(all_finite_values_list)
        if all_finite_values.size > 0:
            vmin_calc, vmax_calc = np.min(all_finite_values), np.max(all_finite_values)
            if vmin_calc == vmax_calc:  # Handle single unique value
                vmin_calc -= 0.5 if vmin_calc != 0 else -0.5
                vmax_calc += 0.5 if vmax_calc != 0 else 0.5

    norm = mcolors.Normalize(vmin=vmin_calc, vmax=vmax_calc)
    cmap_obj = plt.get_cmap(cmap_name)
    cmap_obj.set_bad(color="lightgrey")

    # --- GridSpec Ratios and Figure Size Logic ---
    cbar_fraction_of_tile_width = 0.6  # Defines cbar width relative to one tile width

    # Initialize ratios for GridSpec
    height_ratios_to_use = []
    width_ratios_to_use = []

    current_fig = fig
    if current_fig is None:  # We are creating the figure
        # Determine figure margins first, as they affect figsize calculation if tile size is fixed
        current_fig_margins = {"left": 0.15, "right": 0.90, "bottom": 0.15, "top": 0.92}
        if fig_creation_margins:
            current_fig_margins.update(fig_creation_margins)

        max_x_lbl_len = len(max(x_labels, key=len, default="")) if x_labels else 0
        if max_x_lbl_len > 3:
            needed_bottom_margin = 0.10 + max_x_lbl_len * 0.018
            current_fig_margins["bottom"] = max(
                current_fig_margins.get("bottom", 0.15), min(0.45, needed_bottom_margin)
            )

        max_y_grp_lbl_len = 0
        if y_label_groups:
            max_y_grp_lbl_len = len(
                max((lg[0] for lg in y_label_groups if lg[0]), key=len, default="")
            )
        if max_y_grp_lbl_len > 3:
            needed_left_margin = 0.10 + max_y_grp_lbl_len * 0.012
            current_fig_margins["left"] = max(
                current_fig_margins.get("left", 0.15), min(0.45, needed_left_margin)
            )

        if desired_tile_in is not None:
            tile_w_in, tile_h_in = desired_tile_in
            if tile_w_in <= 0 or tile_h_in <= 0:
                raise ValueError("desired_tile_in dimensions must be positive.")

            height_ratios_to_use = [num_rows * tile_h_in, tile_h_in]  # Physical heights for ratios
            width_ratios_to_use = [
                tile_w_in,
                num_cols * tile_w_in,
                cbar_fraction_of_tile_width * tile_w_in,
            ]  # Physical widths for ratios

            content_drawable_w = sum(width_ratios_to_use)
            content_drawable_h = sum(height_ratios_to_use)

            lm = current_fig_margins["left"]
            rm = current_fig_margins["right"]
            bm = current_fig_margins["bottom"]
            tm = current_fig_margins["top"]

            fig_width_drawable_fraction = max(0.1, rm - lm)
            fig_height_drawable_fraction = max(0.1, tm - bm)

            actual_figsize_w = content_drawable_w / fig_width_drawable_fraction
            actual_figsize_h = content_drawable_h / fig_height_drawable_fraction
            actual_figsize = (actual_figsize_w, actual_figsize_h)
        else:  # desired_tile_in is None, use relative ratios and figsize argument or estimation
            height_ratios_to_use = [float(max(1, num_rows)), 1.0]
            width_ratios_to_use = [1.0, float(max(1, num_cols)), cbar_fraction_of_tile_width]

            actual_figsize = figsize  # Use provided figsize
            if actual_figsize is None:  # Estimate figsize if not provided
                base_fig_w, base_fig_h = 7, 6
                est_width = (
                    base_fig_w
                    + num_cols * 0.45
                    + max_y_grp_lbl_len * 0.11
                    + width_ratios_to_use[0] * 0.2
                    + width_ratios_to_use[2] * 0.2
                )
                est_height = (
                    base_fig_h
                    + num_rows * 0.45
                    + max_x_lbl_len * 0.11
                    + height_ratios_to_use[1] * 0.2
                )
                actual_figsize = (max(10, est_width), max(8, est_height))

        current_fig = plt.figure(figsize=actual_figsize)
        current_fig.subplots_adjust(**current_fig_margins)
        base_spec_for_plot = current_fig.add_gridspec(1, 1)[0, 0]
    else:  # Using provided fig and subplot_spec
        if subplot_spec is None:
            raise ValueError("If 'fig' is provided, 'subplot_spec' must also be provided.")
        base_spec_for_plot = subplot_spec
        if (
            desired_tile_in is not None
        ):  # Set ratios based on desired tile dimensions for proportion
            tile_w_in, tile_h_in = desired_tile_in
            if tile_w_in <= 0 or tile_h_in <= 0:
                raise ValueError("desired_tile_in dimensions must be positive.")
            height_ratios_to_use = [num_rows * tile_h_in, tile_h_in]
            width_ratios_to_use = [
                tile_w_in,
                num_cols * tile_w_in,
                cbar_fraction_of_tile_width * tile_w_in,
            ]
        else:  # desired_tile_in is None, use relative ratios
            height_ratios_to_use = [float(max(1, num_rows)), 1.0]
            width_ratios_to_use = [1.0, float(max(1, num_cols)), cbar_fraction_of_tile_width]

    gs_params_for_inner_layout = {
        "height_ratios": height_ratios_to_use,
        "width_ratios": width_ratios_to_use,
        "wspace": 0.04,
        "hspace": 0.01,  # Minimal space for tight alignment
    }

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=base_spec_for_plot, **gs_params_for_inner_layout
    )

    # --- Axes Definition ---
    ax_row_avg = current_fig.add_subplot(inner_gs[0, 0])
    ax_heatmap = current_fig.add_subplot(inner_gs[0, 1], sharey=ax_row_avg)
    ax_cbar = current_fig.add_subplot(inner_gs[0, 2])
    ax_grand_avg = current_fig.add_subplot(inner_gs[1, 0])
    ax_col_avg = current_fig.add_subplot(inner_gs[1, 1], sharex=ax_heatmap)

    # --- Y-axis Group Label Processing & Drawing ---
    group_y_tick_positions = []
    group_y_tick_labels = []
    current_row_idx_counter = 0
    y_coords_for_separator_lines = []

    if y_label_groups:
        for group_name, count in y_label_groups:
            if count <= 0:
                raise ValueError(f"Group count must be positive: {count} for '{group_name}'.")
            start_row, end_row = current_row_idx_counter, current_row_idx_counter + count - 1
            group_y_tick_positions.append((start_row + end_row) / 2.0)
            group_y_tick_labels.append(str(group_name))
            current_row_idx_counter += count
            if current_row_idx_counter < num_rows:
                y_coords_for_separator_lines.append(end_row + 0.5)

    # --- Row Average Strip (Left) with Grouped Y-Labels ---
    ax_row_avg.imshow(row_means, cmap=cmap_obj, norm=norm, aspect="auto", interpolation="nearest")
    if y_label_groups:
        ax_row_avg.set_yticks(group_y_tick_positions)
        ax_row_avg.set_yticklabels(group_y_tick_labels)
    else:  # Fallback for empty y_label_groups (should be caught by num_rows check)
        ax_row_avg.set_yticks([])
        ax_row_avg.set_yticklabels([])
    plt.setp(ax_row_avg.get_yticklabels(), va="center")
    ax_row_avg.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_row_avg.tick_params(axis="y", which="major", length=0, pad=5, right=False, direction="out")

    # --- Main Heatmap (Center) ---
    im_heatmap = ax_heatmap.imshow(
        data, cmap=cmap_obj, norm=norm, aspect="auto", interpolation="nearest"
    )
    plt.setp(ax_heatmap.get_xticklabels(), visible=False)
    plt.setp(ax_heatmap.get_yticklabels(), visible=False)
    ax_heatmap.tick_params(axis="both", which="both", length=0)
    if title:
        ax_heatmap.set_title(title, loc="center", pad=15)

    for y_coord in y_coords_for_separator_lines:  # Dotted lines for Y-groups
        ax_heatmap.axhline(y_coord, color="black", linestyle=":", linewidth=2)

    # --- Column Average Strip (Bottom) ---
    ax_col_avg.imshow(col_means, cmap=cmap_obj, norm=norm, aspect="auto", interpolation="nearest")
    ax_col_avg.set_xticks(np.arange(num_cols))
    ax_col_avg.set_xticklabels([str(lbl) for lbl in x_labels])
    plt.setp(ax_col_avg.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax_col_avg.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax_col_avg.tick_params(axis="x", which="both", top=False, direction="out")

    # --- Grand Average Cell (Bottom-Left corner) ---
    ax_grand_avg.imshow(
        np.array([[grand_mean_val]]),
        cmap=cmap_obj,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax_grand_avg.tick_params(
        axis="both", which="both", length=0, labelbottom=False, labelleft=False
    )
    if np.isnan(grand_mean_val):
        ax_grand_avg.text(
            0.5,
            0.5,
            "NaN",
            ha="center",
            va="center",
            color="grey",
            fontsize="small",
            transform=ax_grand_avg.trans_axes,
        )
    else:
        text_color = _get_text_color_for_background(cmap_obj(norm(grand_mean_val))[:3])
        ax_grand_avg.text(
            0.5,
            0.5,
            f"{grand_mean_val:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize="small",
            transform=ax_grand_avg.trans_axes,
        )

    # --- Colorbar ---
    current_fig.colorbar(im_heatmap, cax=ax_cbar, orientation="vertical", label="Value")

    return current_fig, (ax_heatmap, ax_row_avg, ax_col_avg, ax_grand_avg, ax_cbar)
