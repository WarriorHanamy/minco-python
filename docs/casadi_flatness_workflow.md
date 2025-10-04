# CasADi Flatness Development Quickstart

The Python extension ships both the native C++ flatness map and the
CasADi-generated variant (`minco.flatness.CasadiQuadrotorFlatnessMap` and
`minco.gcopter.GCOPTERPolytopeSFCCasadi`).

This walkthrough shows the minimal loop for tweaking the CasADi-powered
quadrotor flatness map, rebuilding the generated code, and validating the
result inside the Python bindings.

## 1. Edit the forward mapping

Update the symbolic definition in
`src/casadi_flatness/quadrotor_flatness_generator.py`. For example, to scale the
computed thrust for experimentation you could temporarily change:

```python
thrust = z0 * f_term0 + z1 * f_term1 + z2 * f_term2
```

to:

```python
thrust = 1.01 * (z0 * f_term0 + z1 * f_term1 + z2 * f_term2)
```

Save the file once the desired modification is in place.

## 2. Regenerate the CasADi C sources

```bash
uv run tools/build_casadi_flatness
```

This command rebuilds `src/minco_trajectory/src/casadi_generated/quadrotor_flatness.c`
and the matching headers so the compiled extension picks up your symbolic change.

## 3. Rebuild the pybind11 extension

```bash
uv run tools/rebuild_extension
```

Running the editable install step ensures the newly generated CasADi code is
compiled into the `minco` extension.

## 4. Validate with the GCOPTER CasADi test

```bash
uv run pytest -k gcopter_casadi_flatness
```

The test suite compares the CasADi-backed optimizer against the native
flatness implementation. If your modification changes dynamics, expect this test
to fail—use it as a guardrail while iterating.
