# Synaxis

Synaxis is a model based design framework for building and simulating complex systems. It provides an object-oriented interface for defining system components and their connections. Those systems can then be compiled to a graph representation which allows for pure functional simulation of the system as is or with your favorite numerical solver.

Synaxis is designed to be flexible and extensible, allowing users to easily define new component types and simulation methods.


### Installation

```
pip install synaxis
```

## Roadmap
General rule:
> Every signal should have **kind** (source, output, internal, coutinuous state, discrete state, algebraic unknown, residual, Wiener, time, clock) and every non-source signal should have exactly one definition (equation, connection, residual).

We aim for minimal package specific code from the user perspective. Hence, a vocabulary that could be described like: `<kinds>.<verbs>.<triggers>` seems to reduce the needs to learn a collection of base classes for each kind of system.

- [ ] Sources signals (Param, Input, State) read the nominal value (default to `0.0`) during tracing. This make shape dependent code requiring the definition of `nominal_value` attribute or the usage of helpers like `_matvec` to avoid compilation error.
- [ ] Filter recorded signals to reduce memory usage (or record a subset of signals - lower rate sampling)
- [x] ODE: `dx = f(x,u,t)`; expose `(x, dx)` pairs; with `@x.derivative`
- [ ] SDE: `dx = f dt + g dW` (Itô); expose `(x, f, g)` triplet and Wiener dims; with `@x.diffusion`
- [ ] PDE: `arrays + boundary conditions`; expose shapes and boundary conditions as constraints; with same as ODE + shape metadata but with boundary conditions via residuals
- [ ] Discrete / Multi-rate: `x_{t+1} = f(x,u)`; expose `(x, x_{t+1}, clock)`; with `x = DiscretState(on=clock)` and `@x.update`
- [ ] DAE / Implicit solver: `0 = g(x, z)`; expose unknows (`z`) with residuals; with `Algebraic` and `@Residual` ?
- [ ] Events: guards + reset; expose zero-crossing functions and reset maps; with `@x.update(on=crossing(...))` ?
- [ ] `scipy.integrate.solve_ivp` adapter
- [ ] `diffrax` adapter
- [ ] Noise subclasses
- [ ] Per-step PRNG key source that equation can split and consume lazily instead of per-step
- [ ] CompileSystem immutability while keeping user interface features
- [ ] Euler-Maruyama solver
