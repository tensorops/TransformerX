from itertools import cycle
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class Plot:
    def plot_pe(
            self,
            cols: Tuple[int, list, np.ndarray],
            pos_encodings,
            num_steps,
            show_grid=True,
    ):
        ax = plt.figure(figsize=(6, 2.5), dpi=1000)

        lines = ["-", "--", "-.", ":"]
        self.line_cycler = cycle(lines)

        def plot_line(col):
            plt.plot(
                    np.arange(num_steps),
                    pos_encodings[0, :, col].T,
                    next(self.line_cycler),
                    label=f"col {col}",
            )

        if isinstance(cols, (list, np.ndarray)):
            for col in cols:
                plot_line(col)
        else:
            plot_line(cols)
        ax.legend()
        plt.title("Columns 7-10")
        plt.grid(show_grid)
        plt.show()
