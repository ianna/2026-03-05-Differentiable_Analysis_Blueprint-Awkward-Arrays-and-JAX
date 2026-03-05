# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import awkward as ak
import numpy as np
import psutil
import os
import time
import sys
import ctypes
from juliacall import Main as jl

# --- 1. UTILITIES ---
def get_mem():
    """Returns Resident Set Size (RSS) in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

# --- 2. JULIA ENGINE INITIALIZATION ---
print(f"[{get_mem():.1f} MB] 🔧 Initializing Julia Engine (Zygote)...")
jl.seval("using Zygote")

# We define the 'Safe' Scoped Hessian. 
# It processes event-by-event to keep the AD tape tiny.
jl.seval("""
function ragged_second_deriv_safe(offsets, content)
    # The Physics Model: Sum of squares for a single event
    event_model(c_sub) = sum(abs2, c_sub)

    # Pre-allocate result buffer in Julia (Zero-Copy back to Python later)
    f2_results = zeros(Float64, length(content))
    
    # Ragged-First Scoping: Loop over events manually
    for i in 1:(length(offsets)-1)
        s = offsets[i] + 1
        e = offsets[i+1]
        if s <= e
            # Extract a VIEW (not a copy) of the event
            slice_view = view(content, s:e)
            
            # Compute Diagonal Hessian for THIS event only
            # This prevents the 1.4GB global tape explosion
            h = Zygote.diaghessian(event_model, slice_view)[1]
            f2_results[s:e] .= h
        end
    end
    return f2_results
end
""")

# --- 3. DATA PREPARATION (100k Particles) ---
n_particles = 100_000
print(f"[{get_mem():.1f} MB] 📦 Generating Jagged Physics Data...")
# Simulate 1000 events with 100 particles each
raw_data = np.random.rand(n_particles).astype(np.float64)
offsets_data = np.arange(0, n_particles + 1, 100).astype(np.int64)

# --- 4. WARM-UP (JIT Compilation) ---
# Running a 2-element sample so the audience doesn't wait for LLVM during the demo
print(f"[{get_mem():.1f} MB] 🔥 Warming up JIT...")
_ = jl.ragged_second_deriv_safe(np.array([0, 2], dtype=np.int64), 
                                np.array([1.0, 2.0], dtype=np.float64))

# --- 5. EXECUTION: THE SCOPED HESSIAN ---
mem_before = get_mem()
print(f"[{mem_before:.1f} MB] 🚀 Executing Scoped Hessian (f'')...")
start_time = time.perf_counter()

# THE BRIDGE CALL
# result_f2 will be a native Julia array wrapped as a PyArray
result_f2 = jl.ragged_second_deriv_safe(offsets_data, raw_data)

latency_ms = (time.perf_counter() - start_time) * 1000
mem_after = get_mem()

# --- 6. CLEAN OUTPUT & VERIFICATION ---
print("\n" + "="*40)
print(f"✅ HESSIAN SUCCESS (Ragged-First)")
print(f"⏱️  LATENCY: {latency_ms:.2f} ms")
print(f"📊 MEMORY DELTA: {mem_after - mem_before:.2f} MB")
print(f"🔢 VERIFICATION (f'' should be 2.0): {result_f2[0]:.4f}")
print("="*40)

# --- 7. HARDWARE-AWARE CLEANUP ---
print(f"\n[{get_mem():.1f} MB] 🧹 Releasing memory...")
del result_f2
jl.GC.gc()

if sys.platform.startswith('linux'):
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        print("✨ Linux Memory Trimmed.")
    except: pass
else:
    print(f"ℹ️  Manual trim skipped on {sys.platform}.")

print(f"[{get_mem():.1f} MB] 🎯 Final State.")

# Force exit to prevent the Python/Julia GC race condition 'Bus Error'
os._exit(0)
