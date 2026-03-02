import awkward as ak
from juliacall import Main as jl

# 1. Setup
jl.seval("using AwkwardArray, ForwardDiff")

# 2. Differentiate inside the map
# We map the jacobian over the data
jl.seval("""
function get_ragged_jacobian(p, data)
    # For each sub-list 'x' in our ragged data:
    return map(data) do x
        # Compute the jacobian of the math w.r.t 'p' for this specific sub-list
        ForwardDiff.jacobian(p_ -> x .* p_[1] .+ p_[2], p)
    end
end
""")

# 3. Data & Params
data_py = ak.Array([[10.1, 20.2], [], [30.3]])
params_py = [1.5, 2.0, 3.5]

# 4. Conversion (zero-copy)
jl_data = jl.AwkwardArray.Array(data_py)
jl_params = jl.convert(jl.Vector[jl.Float64], params_py)

# 5. Call our specialized function
# This returns a list of matrices (one matrix per sub-list)
jac_jl = jl.get_ragged_jacobian(jl_params, jl_data)

# 6. Back to Python (copy) (for zero-copy -- see AwkwardArray.jl tutorial)
jac_py = ak.from_iter(jac_jl)
print("--- Final Ragged Jacobian ---")
print(jac_py)
