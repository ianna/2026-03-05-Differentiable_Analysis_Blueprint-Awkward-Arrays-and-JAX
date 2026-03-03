import numpy as np
import psutil
import os
import time
from juliacall import Main as jl

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

# --- 1. JULIA ENGINE ---
print(f"[{get_mem():.1f} MB] 🔧 Initializing Julia (Naive Mode)...")
jl.seval("using Zygote")

# This kernel is 'Naive' because it doesn't loop. 
# It asks for the Hessian of the entire content buffer in one go.
jl.seval("""
function naive_global_hessian(offsets, content)
    # The 'Global' Objective Function
    # Zygote must track every single operation in one massive 'tape'
    function global_model(c)
        total = 0.0
        for i in 1:(length(offsets)-1)
            s = offsets[i] + 1
            e = offsets[i+1]
            if s <= e
                total += sum(abs2, view(c, s:e))
            end
        end
        return total
    end

    # This is the 'RAM Elephant' trigger: 
    # A second-order derivative on a 100k-element vector
    return Zygote.diaghessian(global_model, content)[1]
end
""")

# --- 2. DATA PREPARATION ---
n_particles = 100_000
raw_data = np.random.rand(n_particles).astype(np.float64)
offsets_data = np.arange(0, n_particles + 1, 100).astype(np.int64)

# --- 3. EXECUTION ---
mem_before = get_mem()
print(f"[{mem_before:.1f} MB] ⚠️ WARNING: Executing Global Hessian (100k particles)...")
start_time = time.perf_counter()

try:
    # This is where the 1.4GB spike and potential crash happen
    result = jl.naive_global_hessian(offsets_data, raw_data)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"✅ COMPLETED (Surprising!) | Latency: {latency_ms:.2f} ms")
    print(f"📊 MEMORY DELTA: {get_mem() - mem_before:.2f} MB")
    
except Exception as e:
    print(f"❌ CRASHED as expected: {e}")

print(f"[{get_mem():.1f} MB] 🎯 Final State.")

