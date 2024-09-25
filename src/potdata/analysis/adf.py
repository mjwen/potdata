"""Compute the angular distribution function (ADF) of one or more configurations."""

from collections.abc import Iterable

import numpy as np
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.neighborlist import build_neighbor_list

from potdata.schema.datapoint import DataCollection, DataPoint


class ADF:
    """
    Class to compute the angular distribution function (ADF).

    The ADF is normalized as a probability density function such that its integral is 1.

    TODO: Add support for multiple elements in the ADF calculation.
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
            raise ValueError("Unknown data type to compute ADF.")

        self.angles = None
        self.adf = None
        self.angle_in_degrees = True

    def compute(
        self,
        elements: tuple[str, str, str],
        rmax: float = 5.0,
        nbins: int = 100,
        max_angle: float = None,
        angle_in_degree: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the ADF.

        Args:
            rmax: Maximum bond to consider as a bond for the ADF calculation.
            nbins: Number of bins to use for the ADF calculation.
            elements: Elements to consider for the ADF calculation. This allows for the
                calculation of the ADF of angles between specific elements. It should be
                given as a tuple of three elements (A, B, C), where A, B, and C are the
                chemical elements of the atoms forming the angle A-B-C, with atom B at
                the center. Each of A, B, and C can None, meaning that all elements are
                considered.
            max_angle: Maximum angle to consider for the ADF calculation. e.g. np.pi.
                If None, the maximum value of angles is used.
            angle_in_degree: Whether to return the angles in degrees. If False, the
                returned angles are in radians.

        Returns:
            angles: 1D angle array for the ADF calculation.
            adf: 1D angular distribution function array.
        """

        # create a neighbor list for each image
        # build_neighbor_list requires cutoffs be provided as the radius of each atom.
        # Here, we set rmax/2 as the radius for each atom, and set the skin to 0.0.
        # This effectively sets the cutoff to rmax.
        nl = [
            build_neighbor_list(
                img,
                cutoffs=[rmax / 2] * len(img.positions),
                skin=0.0,
                self_interaction=False,
            )
            for img in self.images
        ]
        ana = Analysis(images=self.images, nl=nl)

        # get the angles in degrees
        A, B, C = elements
        indices = ana.get_angles(A, B, C, unique=True)
        angles = ana.get_values(indices)

        # flatten the list of angles for all images
        angles = np.concatenate(angles)

        if max_angle is None:
            max_angle = np.max(angles)

        # note, the size of adf is bins, and the size of bin_edges is bins+1,
        adf, bin_edges = np.histogram(angles, bins=nbins, range=(0, max_angle))
        angles = (bin_edges[:-1] + bin_edges[1:]) / 2

        # normalize the ADF
        adf = self._normalize(angles, adf)

        # convert to degrees if needed
        if not angle_in_degree:
            angles = angles / 180.0 * np.pi

        self.adf = adf
        self.angles = angles
        self.angle_in_degrees = angle_in_degree

        return self.angles, self.adf

    @staticmethod
    def _normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Normalize the quantity y such that the integral of y over x is 1.

        Args:
            x: x values to normalize y over. Shape (N,).
            y: y values to normalize. Shape (N,).
        """
        area = np.trapz(y, x)
        normalized_y = y / area

        return normalized_y

    def plot(self, filename="adf.pdf", save=True, show=False):
        """
        Plot the ADF vs angle data.

        Args:
            filename: Name of the file to save the ADF data.
            save: Whether to save the plot to a file.
            show: Whether to show the plot.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.8, 3.6))

        ax.plot(self.angles, self.adf, label="Angular Distribution Function")

        xlabel = "Angle (degree)" if self.angle_in_degrees else "Angle (radian)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ADF")

        if save:
            fig.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()


if __name__ == "__main__":
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms = atoms.repeat((2, 2, 2))

    # Get Na-Cl-Na angles
    # For the roksalt structure (Na-Cl bond length of 2.81), with a cutoff of 3, the
    # resulting Na-Cl-Na angles should be at 90 and 180 degrees.
    adf = ADF(atoms)
    adf.compute(rmax=3, elements=("Na", "Cl", "Na"), angle_in_degree=True)

    # adf.plot()
    # from ase.visualize import view
    # view(atoms)
