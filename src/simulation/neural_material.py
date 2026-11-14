"""Radio material with properties from a neural network.

The method is "Neural Materials" of arXiv:2311.18558, Section III-D. A multilayer
perceptron (MLP) gives the material properties at each point where a ray hits a surface.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import NamedTuple

import drjit as dr
import mitsuba as mi
import numpy as np
import sionna.rt
from drjit import nn

# The Dr.Jit CUDA backend evaluates a packed network only with half-precision weights.
_WEIGHT_FLOAT = dr.float16_array_t(mi.Float)
_WEIGHT_TENSOR = dr.tensor_t(_WEIGHT_FLOAT)
_STATE_KEYS = ("weights", "encoding_octaves", "hidden_layers", "hidden_units", "center", "extent")


class MaterialValues(NamedTuple):
    """Material properties at a set of points. Each field has one value for each point."""

    relative_permittivity: mi.Float
    conductivity: mi.Float
    scattering_coefficient: mi.Float
    xpd_coefficient: mi.Float
    thickness: mi.Float


def make_material_network(
    encoding_octaves: int, hidden_layers: int, hidden_units: int, seed: int
) -> nn.Module:
    """Make the MLP of a `NeuralRadioMaterial`.

    The network has a positional encoding (arXiv:2311.18558, Eq. 36), then the hidden
    layers with ReLU activations, then one linear layer with one output for each field of
    `MaterialValues`.

    Args:
        encoding_octaves: Number of frequencies of the positional encoding.
        hidden_layers: Number of hidden layers.
        hidden_units: Number of neurons in each hidden layer.
        seed: Seed of the initial weights.

    Returns:
        The network. The key `weights` gives all weights as one array.

    Raises:
        ValueError: If `encoding_octaves`, `hidden_layers`, or `hidden_units` is less
            than 1.
    """
    sizes = {
        "encoding_octaves": encoding_octaves,
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
    }
    for name, value in sizes.items():
        if value < 1:
            raise ValueError(f"{name} must be 1 or more, but it is {value}.")
    layers: list[nn.Module] = [nn.SinEncode(encoding_octaves), nn.Cast(_WEIGHT_FLOAT)]
    for _ in range(hidden_layers):
        layers += [nn.Linear(-1, hidden_units), nn.ReLU()]
    layers += [nn.Linear(-1, len(MaterialValues._fields)), nn.Cast(mi.Float)]
    network = nn.Sequential(*layers).alloc(_WEIGHT_TENSOR, 3, rng=dr.rng(seed=seed))
    return nn.pack(network, layout="training")


def _architecture(network: nn.Module) -> tuple[int, int, int]:
    """Return the arguments of `make_material_network` that give this network structure."""
    linear = [layer for layer in network.layers if isinstance(layer, nn.Linear)]
    encoding = next(layer for layer in network.layers if isinstance(layer, nn.SinEncode))
    return encoding.octaves, len(linear) - 1, linear[0].config[1]


def _sigmoid(value: mi.Float) -> mi.Float:
    return dr.rcp(1.0 + dr.exp(-value))


class NeuralRadioMaterial(sionna.rt.RadioMaterial):
    """Radio material whose properties are a function of the position.

    The class uses the field calculations of `sionna.rt.RadioMaterial`. Only the source
    of the material properties is different: the network gives them at the point of each
    ray interaction.

    Args:
        name: Name of the material.
        network: A network from `make_material_network`.
        center: Center of the bounding box of the scene, in meters.
        extent: Length of the longest side of the bounding box of the scene, in meters.
    """

    def __init__(self, name: str, network: nn.Module, center: mi.Point3f, extent: float):
        super().__init__(name=name)
        self.network = network
        self._center = mi.Point3f(center)
        self._extent = float(extent)

    def values_at(self, points: mi.Point3f | np.ndarray) -> MaterialValues:
        """Calculate the material properties at the given points.

        Args:
            points: Positions in meters. A NumPy array must have the shape `[n, 3]`.

        Returns:
            The material properties. The relative permittivity is more than 1. The
            conductivity (S/m) and the thickness (m) are more than 0. The scattering
            coefficient and the XPD coefficient are between 0 and 1.
        """
        if isinstance(points, np.ndarray):
            points = mi.Point3f(np.ascontiguousarray(points.T, dtype=np.float32))
        # Normalization of arXiv:2311.18558, Eq. 37.
        unit = (points - self._center) / self._extent
        permittivity, conductivity, scattering, xpd, thickness = (
            mi.Float(output) for output in self.network(nn.CoopVec(unit.x, unit.y, unit.z))
        )
        return MaterialValues(
            # Output transformations of arXiv:2311.18558, Fig. 1 and Eq. 32 to 35.
            relative_permittivity=1.0 + dr.exp(permittivity),
            conductivity=dr.exp(conductivity),
            scattering_coefficient=_sigmoid(scattering),
            xpd_coefficient=_sigmoid(xpd),
            # Assumption: the paper has no thickness. The exponential keeps it positive.
            thickness=dr.exp(thickness),
        )

    @contextmanager
    def _values_of(self, si: mi.SurfaceInteraction3f) -> Iterator[None]:
        """Set the properties that the base class reads to the values at `si.p`."""
        saved = (self._eta_r, self._sigma, self._s, self._kx, self._d)
        values = self.values_at(si.p)
        self._eta_r = values.relative_permittivity
        self._sigma = values.conductivity
        self._s = values.scattering_coefficient
        self._kx = values.xpd_coefficient
        self._d = values.thickness
        try:
            yield
        finally:
            self._eta_r, self._sigma, self._s, self._kx, self._d = saved

    def sample(self, ctx, si, sample1, sample2, active=True):
        """Sample an interaction with the properties at the interaction point."""
        with self._values_of(si):
            return super().sample(ctx, si, sample1, sample2, active)

    def eval(self, ctx, si, wo, active=True):
        """Evaluate the Jones matrix with the properties at the interaction point."""
        with self._values_of(si):
            return super().eval(ctx, si, wo, active)

    def pdf(self, ctx, si, wo, active=True):
        """Give the probability of an interaction with the properties at the interaction point."""
        with self._values_of(si):
            return super().pdf(ctx, si, wo, active)

    def state(self) -> dict[str, np.ndarray]:
        """Return the data that `from_state` needs to make an equal material.

        Returns:
            The weights, the network structure (`encoding_octaves`, `hidden_layers`,
            `hidden_units`), and the normalization (`center`, `extent`).
        """
        encoding_octaves, hidden_layers, hidden_units = _architecture(self.network)
        return {
            "weights": np.array(self.network["weights"], dtype=np.float16),
            "encoding_octaves": np.array(encoding_octaves),
            "hidden_layers": np.array(hidden_layers),
            "hidden_units": np.array(hidden_units),
            "center": np.ravel(self._center.numpy()),
            "extent": np.array(self._extent),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray], name: str) -> "NeuralRadioMaterial":
        """Make a material from the result of `state`.

        Args:
            state: The result of `state`.
            name: Name of the material.

        Returns:
            The material.

        Raises:
            ValueError: If a key is missing, or if the number of weights does not agree
                with the network structure.
        """
        missing = [key for key in _STATE_KEYS if key not in state]
        if missing:
            raise ValueError(f"The state does not have the keys {missing}.")
        network = make_material_network(
            int(state["encoding_octaves"]),
            int(state["hidden_layers"]),
            int(state["hidden_units"]),
            seed=0,
        )
        weights = np.ravel(state["weights"])
        expected = dr.width(network["weights"])
        if weights.size != expected:
            raise ValueError(
                f"The network structure needs {expected} weights, but the state has {weights.size}."
            )
        network["weights"] = _WEIGHT_FLOAT(weights.astype(np.float16))
        center = mi.Point3f([float(value) for value in np.ravel(state["center"])])
        return cls(name, network, center, float(state["extent"]))


def assign_material(scene: sionna.rt.Scene, material: NeuralRadioMaterial) -> None:
    """Give the material to all objects of the scene.

    Args:
        scene: The scene.
        material: The material.
    """
    for scene_object in scene.objects.values():
        scene_object.radio_material = material


def apply_neural_material(scene: sionna.rt.Scene, network: nn.Module) -> NeuralRadioMaterial:
    """Make one `NeuralRadioMaterial` for the scene and give it to all objects.

    The bounding box of the scene gives the normalization of the positions.

    Args:
        scene: The scene.
        network: A network from `make_material_network`.

    Returns:
        The material.
    """
    bbox = scene.mi_scene.bbox()
    center = mi.Point3f((bbox.min + bbox.max) / 2)
    extent = float(dr.max(bbox.max - bbox.min))
    material = NeuralRadioMaterial("neural", network, center, extent)
    assign_material(scene, material)
    return material
