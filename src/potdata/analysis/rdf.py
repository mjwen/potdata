"""Compute the radial distribution function (RDF) of one or more configurations."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry.analysis import Analysis

from potdata.schema.datapoint import DataCollection, DataPoint


class RDF:
    """
    Compute the radial distribution function (RDF) of a configuration or a trajectory
    of configurations.

    Args:
        data: configuration data. Can be either a single ASE Atoms, a trajectory of
        ASE Atoms objects, a DataPoint, or a DataCollection.
    """

    def __init__(self, data: Atoms | list[Atoms] | DataPoint | DataCollection):
        if isinstance(data, Atoms):
            self.images = [data]
        elif isinstance(data, DataPoint):
            self.images = [data.to_ase_atoms()]
        elif isinstance(data, DataCollection):
            self.images = [dp.to_ase_atoms() for dp in data]
        elif isinstance(data, Iterable) and isinstance(list(data)[0], Atoms):
            self.images = data  # type: ignore
        else:
            raise ValueError("Unknown data type to compute RDF.")

        self.dist = None
        self.rdf = None

    def compute(
        self, rmax: float = 5.0, nbins: int = 100, elements=None, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the RDF.

        Args:
            rmax: Maximum distance to consider for the RDF calculation.
            nbins: Number of bins to use for the RDF calculation.
            elements: Elements to consider for the RDF calculation. This allows for the
                calculation of the RDF for specific element pairs. If None, all elements
                are considered. See ase.geometry.analysis.Analysis.get_rdf for more.
            kwargs: Additional arguments to ase.geometry.analysis.Analysis.get_rdf.

        Returns:
            dist: 1D distance array for the RDF calculation.
            rdf: 1D radial distribution function array.
        """

        ana = Analysis(images=self.images)

        out = ana.get_rdf(
            rmax=rmax, nbins=nbins, elements=elements, return_dists=True, **kwargs
        )

        # rdf for individual configs
        rdf = [pair[0] for pair in out]

        # averaged data
        self.rdf = np.mean(rdf, axis=0)
        # distance for every config is the same
        self.dist = np.asarray(out[0][1])

        return self.dist, self.rdf

    def save_data(self, filename: str = "rdf.json"):
        """
        Save the RDF vs distance data to a JSON file.

        Args:
            filename: Name of the file to save the RDF data.
        """

        df = pd.DataFrame({"distance": self.dist, "rdf": self.rdf})
        df.to_json(filename)

    def plot(self, filename: str = "rdf.pdf", save: bool = True, show: bool = False):
        """
        Plot the RDF vs distance data.

        Args:
            filename: Name of the file to save the plot.
            save: If True, the plot will be saved to the file.
            show: If True, the plot will be displayed.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.8, 3.6))

        # plot the RDF
        ax.plot(self.dist, self.rdf, color="C1")
        ax.set_xlabel("Distance")
        ax.set_ylabel("RDF")

        # plot the line at y=1
        ax.plot(ax.get_xlim(), [1, 1], "--", color="gray")

        if save:
            fig.savefig(filename, bbox_inches="tight")

        if show:
            plt.show()


if __name__ == "__main__":
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms = atoms.repeat((4, 4, 4))

    # get Na-Cl pair RDF
    rdf = RDF(atoms)
    rdf.compute(elements=["Na", "Cl"])
    rdf.plot(show=True)
