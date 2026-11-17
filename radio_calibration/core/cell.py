"""The cell record."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """One sector of a site. Sionna RT simulates each cell as one transmitter.

    Attributes:
        name: Unique name of the cell.
        site: Index of the site that holds the cell.
        x: Position on the x axis of the scene, in meters.
        y: Position on the y axis of the scene, in meters.
        z: Position on the z axis of the scene, in meters.
        azimuth_deg: Rotation about the z axis, in degrees. 0 points to +x.
        downtilt_deg: Tilt below the horizon, in degrees.
        power_rs_dbm: Reference-signal power for each resource element, in dBm.
        num_rows: Number of rows of the planar array.
        num_cols: Number of columns of the planar array.
        vertical_spacing: Distance between rows, in wavelengths.
        horizontal_spacing: Distance between columns, in wavelengths.
        pattern: Name of the Sionna RT antenna pattern.
        polarization: Name of the Sionna RT polarization.
    """

    name: str
    site: int
    x: float
    y: float
    z: float
    azimuth_deg: float
    downtilt_deg: float
    power_rs_dbm: float
    num_rows: int
    num_cols: int
    vertical_spacing: float
    horizontal_spacing: float
    pattern: str
    polarization: str
