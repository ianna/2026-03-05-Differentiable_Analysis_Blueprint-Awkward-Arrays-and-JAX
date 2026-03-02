# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import awkward as ak
from juliacall import Main as jl
import matplotlib.pyplot as plt

# 1. Setup Julia AD Engine
jl.seval("using ForwardDiff, AwkwardArray")

# Our Physics Model: Quadratic Trajectory y = v0*x + 0.5*a*x^2
# p = [v0, a]
jl.seval("""
function trajectory_model(x, p)
    return p[1] .* x .+ 0.5 .* p[2] .* (x .^ 2)
end

function compute_ragged_jacobians(data_ragged, params)
    # ForwardDiff.jacobian maps over the ragged events
    return map(data_ragged) do event_hits
        ForwardDiff.jacobian(p -> trajectory_model(event_hits, p), params)
    end
end
""")

# 2. SIMULATE REALISTIC RAGGED DATA (Python Side)
# We simulate a few "Events" in a tracker.
np.random.seed(42)
num_events = 100
params_true = [10.0, -9.8]  # v0=10 m/s, a=-9.8 m/s^2 (gravity)

# Generate variable numbers of hits per event (poisson distributed)
hits_per_event = np.random.poisson(lam=8, size=num_events)

# Create ragged lists of hit positions (time 'x' in this case)
data_list = []
for n in hits_per_event:
    # Generate random hit times, sorted for physics sanity
    hit_times = np.sort(np.random.uniform(0.1, 2.0, size=n))
    data_list.append(hit_times)

# Wrap in an Awkward Array
x_hits_ragged = ak.Array(data_list)

# Compute the "Measured" y-values using our true parameters (with noise)
# Model is vectorized by Awkward!
y_measured_ragged = x_hits_ragged * params_true[0] + 0.5 * params_true[1] * (x_hits_ragged ** 2)
# Add detector resolution noise
y_noise = ak.Array([np.random.normal(0, 0.5, size=n) for n in hits_per_event])
y_measured_ragged = y_measured_ragged + y_noise

# 3. CONVERT TO JULIA (ZERO-COPY)
# We hand the ragged x-positions to Julia to compute Jacobians
jl_x_hits = jl.AwkwardArray.Array(x_hits_ragged)
jl_params = jl.convert(jl.Vector[jl.Float64], params_true)

# 4. COMPUTE JACOBIANS IN JULIA
# jac_jl is a Julia Vector of Matrices (one matrix per event)
jac_jl = jl.compute_ragged_jacobians(jl_x_hits, jl_params)

# ---------------------------------------------------------
# 5. ZERO-COPY BRIDGE & VERIFICATION
# ---------------------------------------------------------
# Create NumPy views (Zero-Copy)
py_np_views = [np.array(m, copy=False) for m in jac_jl]

# Assemble the final '3D' Jacobian Array (consolidation copy happens here)
# Type will be: (num_events, var_hits, 2_params)
jac_py = ak.to_regular(ak.Array(py_np_views), axis=2)

# Verification
jl_ptr = jac_jl[0].__array_interface__['data'][0]
py_ptr = py_np_views[0].ctypes.data
print(f"Zero-Copy Valid? {jl_ptr == py_ptr} ({hex(jl_ptr)})")

# 6. PHYSICS APPLICATION: Error Propagation
# V_p (Parameter Covariance)
V_p = np.diag([0.1**2, 0.05**2]) # 10% unc. in v0, 5% unc. in a

# V_y = J * V_p * J_T. We use the flatten/unflatten pattern.
flat_jac = ak.to_numpy(ak.flatten(jac_py)) # Strategic Copy
JV = flat_jac @ V_p
flat_variances = np.sum(JV * flat_jac, axis=1) # Get variances (diag of V_y)

# Wrap variances back into ragged structure
y_variances_ragged = ak.unflatten(flat_variances, ak.num(jac_py))

# ---------------------------------------------------------
# 7. VISUALIZATION
# ---------------------------------------------------------
# Let's plot the first two events with their AD-propagated errors.
plt.figure(figsize=(12, 6))

for i in range(2):
    plt.subplot(1, 2, i+1)
    
    x = ak.to_numpy(x_hits_ragged[i])
    y = ak.to_numpy(y_measured_ragged[i])
    y_err = np.sqrt(ak.to_numpy(y_variances_ragged[i]))
    
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, ecolor='red', 
                 label=f'Measurements (w/ AD-Errors)')
    
    # Plot smooth true trajectory
    x_fine = np.linspace(0, 2.0, 100)
    y_true = x_fine * params_true[0] + 0.5 * params_true[1] * (x_fine ** 2)
    plt.plot(x_fine, y_true, color='black', linestyle='--', label='True Model')
    
    plt.title(f"Event {i} Tracking: {len(x)} Ragged Hits")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.ylim(-5, 12)
    if i == 0: plt.legend()

plt.tight_layout()
plt.savefig('realistic_ad_plot.png')
print("Successfully generated realistic tracking plot with propagated errors.")
