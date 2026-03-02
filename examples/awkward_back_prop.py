# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import awkward as ak
from juliacall import Main as jl
import matplotlib.pyplot as plt

# --- PRE-FLIGHT CHECK: Ensure Zygote is installed ---
try:
    jl.seval("using Zygote")
except:
    print("Zygote not found. Installing now (this may take a minute)...")
    jl.seval('import Pkg; Pkg.add("Zygote")')
    jl.seval("using Zygote")

jl.seval("using AwkwardArray")

# --- JULIA KERNEL ---
jl.seval("""
function loss_function(p, x_ragged, y_measured)
    total_loss = 0.0
    for i in 1:length(x_ragged)
        y_pred = p[1] .* x_ragged[i] .+ 0.5 .* p[2] .* (x_ragged[i] .^ 2)
        total_loss += sum((y_pred .- y_measured[i]).^2)
    end
    return total_loss
end

function compute_backprop_gradient(p, x_ragged, y_measured)
    # Reverse-mode AD magic happens here
    return Zygote.gradient(p -> loss_function(p, x_ragged, y_measured), p)[1]
end
""")

# --- OPTIMIZATION ---
params_guess = np.array([12.0, -5.0]) 
learning_rate = 0.0005 # Dropped slightly for stability
history = []

jl_x = jl.AwkwardArray.Array(x_hits_ragged)
jl_y = jl.AwkwardArray.Array(y_measured_ragged)

print("Starting Back-prop Optimization...")
for i in range(101):
    jl_p = jl.convert(jl.Vector[jl.Float64], params_guess)
    grad = np.array(jl.compute_backprop_gradient(jl_p, jl_x, jl_y))
    
    params_guess -= learning_rate * grad
    
    if i % 20 == 0:
        loss = jl.loss_function(jl_p, jl_x, jl_y)
        history.append(loss)
        print(f"Iter {i:3}: Loss={loss:8.2f} | v0={params_guess[0]:6.2f} | a={params_guess[1]:6.2f}")

# --- PLOTTING ---
plt.style.use('bmh')
fig, ax = plt.subplots(figsize=(8, 5))

x_fine = np.linspace(0, 1.5, 100)
y_fit = params_guess[0] * x_fine + 0.5 * params_guess[1] * x_fine**2

ax.scatter(ak.flatten(x_hits_ragged[:3]), ak.flatten(y_measured_ragged[:3]), 
           color='black', s=15, alpha=0.4, label='Ragged Hits')
ax.plot(x_fine, y_fit, 'g-', lw=3, label='Back-prop Best Fit')

# Using a raw string (r"") to avoid escape sequence warnings
ax.set_title(r"Global $\chi^2$ Minimization via Back-propagation")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position [m]")
ax.legend()
plt.show()
