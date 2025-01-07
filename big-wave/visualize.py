# Copyright (c) 2024, ETH Zurich

from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import config


def main(base_dir: Path, z: str):
    matplotlib.style.use("ggplot")
    matplotlib.rc("font", family="serif", size=11)
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("figure", figsize=(5, 3))
    matplotlib.rc("legend", fontsize=11)
    matplotlib.rc("axes", linewidth=1)
    matplotlib.rc("lines", linewidth=2)
    plt.tick_params(top="off", right="off", which="both")
    
    sim_dir = base_dir / z
    cfg = config.load(sim_dir / "config.yaml")
    cmp = config.load(sim_dir / "computed.yaml")

    angles = cmp["cutoff_angles"]
    elements = [config.parse_optical_element(el, sim_dir) for el in cfg["elements"]]
    sim_params = config.parse_sim_params(cfg["sim_params"])

    if cfg["multisource"]["type"] != "points":
        raise ValueError("only works with points multisource")

    x_range = cfg["multisource"]["x_range"]

    plt.plot([0, 0], x_range)

    sim_half_width = sim_params.N * sim_params.dx * 0.5

    current_z = 0
    current_max_x = np.max(np.abs(x_range))

    for i, el in enumerate(elements):
        angle = angles[i]
        next_z = el.z_start

        next_max_x = current_max_x + np.tan(angle) * (next_z - current_z)
        plt.plot([current_z, next_z], [current_max_x, next_max_x], color="orange")
        plt.plot([current_z, next_z], [-current_max_x, -next_max_x], color="orange")

        plt.plot([next_z] * 2, [-sim_half_width, sim_half_width], linestyle="dashed", color="darkgreen")

        current_z = next_z
        current_max_x = next_max_x

    angle = angles[-1]
    next_max_x = current_max_x + np.tan(angle) * (sim_params.z_detector - current_z)
    plt.plot([current_z, sim_params.z_detector], [current_max_x, next_max_x], color="orange")
    plt.plot([current_z, sim_params.z_detector], [-current_max_x, -next_max_x], color="orange")

    plt.plot(
        [sim_params.z_detector] * 2,
        [sim_params.detector_size * -0.5, sim_params.detector_size * 0.5],
    )

    plt.xlabel("z [m]")
    plt.ylabel("x [m]")
    plt.tick_params(top="off", right="off", which="both")

    plt.tight_layout()
    plt.savefig(z + ".png")
    plt.show()


for z in [0.1, 0.2, 0.3, 0.5, 1.0, 1.5]:
    main(
        Path(
            "/home/pascal/code/eth/master-thesis/rave-sim/analysis/quadratic_falloff/templates"
        ),
        f"z_{z:.1f}"
    )
