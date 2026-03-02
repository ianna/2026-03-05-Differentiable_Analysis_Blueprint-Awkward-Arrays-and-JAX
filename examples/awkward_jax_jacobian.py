# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import awkward as ak
import jax
import jax.numpy as jnp

# 1. Initialize Awkward on the JAX backend
ak.jax.register_and_check()
params = ak.Array([1.5, 2.0, 3.5], backend="jax") # e.g., MET, b-tag, HT thresholds
data = ak.Array([[10.1, 20.2], [], [30.3]], backend="jax")

def physics_model(p):
    # A dummy ragged calculation: each event scaled by a different parameter
    # In a real case, this would be your Significance or Yield
    return data * p[0] + p[1]

# 2. Define the Manual Jacobian Loop
def get_awkward_jacobian(func, primals):
    n_params = len(primals)
    jacobian_columns = []
    
    for i in range(n_params):
        # Create a tangent vector for the i-th parameter
        # [1, 0, 0], then [0, 1, 0], etc.
        unit_tangent = jnp.zeros(n_params).at[i].set(1.0)
        tangent_ak = ak.Array(unit_tangent, backend="jax")
        
        # Compute JVP: returns (value, derivative_wrt_param_i)
        _, jvp_column = jax.jvp(func, (primals,), (tangent_ak,))
        jacobian_columns.append(jvp_column)
        
    return jacobian_columns

# 3. Execute
jac = get_awkward_jacobian(physics_model, params)
# 'jac' is now a list of Awkward Arrays, each representing a column of the Jacobian:
#
# [<Array [[10.1, 20.2], [], [30.3]] type='3 * var * float64'>, <Array [[1.0, 1.0], [], [1.0]] type='3 * var * float64'>, <Array [[0.0, 0.0], [], [0.0]] type='3 * var * float64'>]
