
API for feature effect:

- `PDP(data, model, [axis_limits, centering])`

`.fit(features, centering)`

`.plot(feature, uncertainty, centering, nof_points)`

`.eval(feature, x, uncertainty, centering)`


Values for certain parameters:

- `features: typing.Union[int, str, list] = "all"`
- `centering: typing.Union[bool, str] = False, string: ["zero_integral", "zero_start"], default: "zero_integral"`
- `axis_limits: typing.Union[None, np.ndarray], default: None`
- `max_nof_instances: typing.Union[int, str], default: 1000`
- `uncertainty: typing.Union[bool, str], default: False, string: ["std", "std_err"], default: "std"`

