"""Class → color palette for trajectory rendering.

Hex strings, no alpha. Colors picked for hue separation on dark
backgrounds. Includes labels seen across the supported dataloader
formats; extend as new label classes appear.
"""

# OSDaR23 (raillabel) class set.
OSDAR23_COLORS: dict[str, str] = {
    "animal":        "#e377c2",
    "bicycle":       "#1f77b4",
    "buffer_stop":   "#8c564b",
    "catenary_pole": "#7f7f7f",
    "drag_shoe":     "#bcbd22",
    "flame":         "#d62728",
    "motorcycle":    "#9467bd",
    "person":        "#2ca02c",
    "road_vehicle":  "#ff7f0e",
    "signal_bridge": "#17becf",
    "signal_pole":   "#6b6ecf",
}

# Delivery-robot (dx3) class set.
DELIVERY_ROBOT_COLORS: dict[str, str] = {
    "human.pedestrian":   "#2ca02c",
    "vehicle":            "#ff7f0e",
    "vehicle.car":        "#ff7f0e",
    "vehicle.truck":      "#d62728",
    "vehicle.bus":        "#bcbd22",
    "vehicle.motorcycle": "#9467bd",
    "vehicle.bicycle":    "#1f77b4",
}

EGO_COLOR = "#00ffff"
DEFAULT_COLOR = "#cccccc"


def color_for(label: str) -> str:
    """Hex color for ``label``, falling back to ``DEFAULT_COLOR``."""
    if label == "ego":
        return EGO_COLOR
    if label in OSDAR23_COLORS:
        return OSDAR23_COLORS[label]
    if label in DELIVERY_ROBOT_COLORS:
        return DELIVERY_ROBOT_COLORS[label]
    return DEFAULT_COLOR
