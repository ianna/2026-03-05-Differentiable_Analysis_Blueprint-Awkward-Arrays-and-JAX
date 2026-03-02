# Differentiable Analysis Blueprint: Awkward Arrays and JAX

High-energy physics (HEP) relies on nested, variable-length (“ragged”) data structures that do not align naturally with the static, rectangular tensor abstractions assumed by most GPU compiler stacks. While Awkward Array provides a NumPy-like interface for such data, integrating it into the JAX ecosystem exposes an architectural mismatch between dynamic, offset-based structures and JAX’s static-shape XLA compilation model.

This presentation evaluates the current state of the Awkward–JAX backend and examines why JAX’s tracing model remains a significant hurdle. The cumulative overhead of tracing, PyTree transformations, compilation latency, and kernel dispatch—the effective “XLA tax”—often negates expected GPU speedups for realistic jagged workloads.

The talk revisits an alternative autodiff architecture based on eager, complex-step differentiation. By leveraging Awkward’s complex-valued kernels and avoiding external tracing systems, this approach could provide near machine-precision forward-mode derivatives that are Numba-compatible and independent of static shape constraints. The central question is whether such an internal, eager approach is a viable long-term path for autodiff in HEP, or whether the architectural mismatch with JAX represents a fundamental barrier for dynamic data.
