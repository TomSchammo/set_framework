from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

Initializer = Union[str, Callable[[torch.Tensor], None], Sequence[float]]


class SReLU(nn.Module):
    """S-shaped Rectified Linear Unit.

    It follows:
    `f(x) = t^r + a^r(x - t^r) for x >= t^r`,
    `f(x) = x for t^r > x > t^l`,
    `f(x) = t^l + a^l(x - t^l) for x <= t^l`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        t_left_initializer: initializer function for the left part intercept
        a_left_initializer: initializer function for the left part slope
        t_right_initializer: initializer function for the right part intercept
        a_right_initializer: initializer function for the right part slope
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    # References
        - [Deep Learning with S-shaped Rectified Linear Activation Units](
           http://arxiv.org/abs/1512.07030)
    """

    def __init__(
        self,
        units: Optional[int] = None,
        t_left_initializer: Initializer = "zeros",
        a_left_initializer: Initializer = (0.0, 1.0),
        t_right_initializer: Initializer = (0.0, 5.0),
        a_right_initializer: Initializer = "ones",
        shared_axes: Optional[Union[int, Iterable[int]]] = None,
        input_shape: Optional[Sequence[Optional[int]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.t_left_initializer = t_left_initializer
        self.a_left_initializer = a_left_initializer
        self.t_right_initializer = t_right_initializer
        self.a_right_initializer = a_right_initializer
        if shared_axes is None:
            self.shared_axes = None
        elif isinstance(shared_axes, (list, tuple)):
            self.shared_axes = list(shared_axes)
        else:
            self.shared_axes = [shared_axes]

        self.t_left: Optional[nn.Parameter] = None
        self.a_left: Optional[nn.Parameter] = None
        self.t_right: Optional[nn.Parameter] = None
        self.a_right: Optional[nn.Parameter] = None
        self._cached_input_shape: Optional[Tuple[Optional[int], ...]] = None

        if input_shape is None and units is not None:
            input_shape = (None, units)

        if input_shape is not None:
            self._cached_input_shape = tuple(input_shape)
            self._build_from_input_shape(tuple(input_shape))

    @property
    def weight(self):
        if self.t_left is None:
            if self._cached_input_shape is None:
                raise RuntimeError(
                    "SReLU parameters are uninitialized. Call the module once "
                    "or pass input_shape to the constructor before accessing "
                    "weight.")
            self._build_from_input_shape(self._cached_input_shape)
        return torch.stack(
            [self.t_left, self.a_left, self.t_right, self.a_right])

    def _apply_initializer(self, tensor: torch.Tensor,
                           initializer: Initializer) -> None:
        if initializer is None:
            return
        if isinstance(initializer, str):
            name = initializer.lower()
            if name == "zeros":
                nn.init.zeros_(tensor)
                return
            if name == "ones":
                nn.init.ones_(tensor)
                return
            if name == "uniform":
                nn.init.uniform_(tensor)
                return
            raise ValueError(f"Unsupported initializer string: {initializer}")
        if isinstance(initializer, (tuple, list)):
            if len(initializer) != 2:
                raise ValueError("Initializer tuple/list must be (min, max).")
            nn.init.uniform_(tensor,
                             a=float(initializer[0]),
                             b=float(initializer[1]))
            return
        if callable(initializer):
            initializer(tensor)
            return
        raise TypeError(f"Unsupported initializer type: {type(initializer)}")

    def _build_from_input_shape(
        self,
        input_shape: Tuple[Optional[int], ...],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if input_shape and input_shape[0] in (None, -1):
            param_shape = list(input_shape[1:])
        else:
            param_shape = list(input_shape)

        if self.shared_axes is not None:
            for axis in self.shared_axes:
                if axis <= 0:
                    raise ValueError(
                        "shared_axes should use 1-based input axes.")
                if axis - 1 >= len(param_shape):
                    raise ValueError(
                        "shared_axes axis is out of input shape range.")
                param_shape[axis - 1] = 1

        param_shape = tuple(int(x) for x in param_shape)
        if any(dim <= 0 for dim in param_shape):
            raise ValueError(f"Invalid input shape for SReLU: {input_shape}")

        t_left = torch.empty(param_shape, device=device, dtype=dtype)
        a_left = torch.empty(param_shape, device=device, dtype=dtype)
        t_right = torch.empty(param_shape, device=device, dtype=dtype)
        a_right = torch.empty(param_shape, device=device, dtype=dtype)

        self._apply_initializer(t_left, self.t_left_initializer)
        self._apply_initializer(a_left, self.a_left_initializer)
        self._apply_initializer(t_right, self.t_right_initializer)
        self._apply_initializer(a_right, self.a_right_initializer)

        self.t_left = nn.Parameter(t_left)
        self.a_left = nn.Parameter(a_left)
        self.t_right = nn.Parameter(t_right)
        self.a_right = nn.Parameter(a_right)

        self.register_parameter("t_left", self.t_left)
        self.register_parameter("a_left", self.a_left)
        self.register_parameter("t_right", self.t_right)
        self.register_parameter("a_right", self.a_right)

    def _ensure_built(self, x: torch.Tensor) -> None:
        if self.t_left is not None:
            return
        self._build_from_input_shape(tuple(x.shape),
                                     device=x.device,
                                     dtype=x.dtype)

    def forward(self, x):
        self._ensure_built(x)
        t_right_actual = self.t_left + torch.abs(self.t_right)

        # shape = [1] * len(x.shape)
        # shape[1] = -1

        t_left = self.t_left
        a_left = self.a_left
        a_right = self.a_right

        relu = torch.where(x - t_left >= 0,
                           torch.minimum(x - t_left, t_right_actual - t_left),
                           a_left * (x - t_left))
        y_left_and_center = t_left + relu
        y_right = torch.relu(x - t_right_actual) * a_right
        return y_left_and_center + y_right
