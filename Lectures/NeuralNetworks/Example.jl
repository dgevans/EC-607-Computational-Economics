### Julia environment setup
using Flux
using Statistics
using Random
using Parameters
using Metal
using Plots
Metal.allowscalar(false)
# If GPU is available, use CUDA for acceleration
#import CUDA
#has_cuda = CUDA.has_cuda()
#if has_cuda
#    println("CUDA detected. Using GPU for training.")
#    CUDA.allowscalar(false)  # disable slow scalar operations on GPU
#end
has_gpu = false
has_cuda = false
has_metal = false

# Set random seed for reproducibility
Random.seed!(42)

### Model parameters for the RBC model (all Float32 for GPU)
@with_kw mutable struct Params
    α::Float32 = 0.30      # capital share in production
    β::Float32 = 0.95      # discount factor
    δ::Float32 = 0.1       # depreciation rate
    γ::Float32 = 2.0       # CRRA utility coefficient (risk aversion)
    ρ::Float32 = 0.90      # persistence of productivity shock
    σ_ε::Float32 = 0.02      # std dev of shock innovation (for log A) 

    #steady state
    k_ss::Float32 = 0.0
    c_ss::Float32 = 0.0
    y_ss::Float32 = 0.0
    A_ss::Float32 = 0.0
end

### Utility function and its marginal derivative
function u(c,γ)
    return c^(1-γ) / (1-γ)
end
function u_prime(c,γ)
    return c.^(-γ)      # derivative of c^(1-γ)/(1-γ) is c^(-γ)
end

# Steady-state (for A=1) for reference (solve α β (A) k^(α-1) + β(1-δ) = 1)
function steady_state(para::Params)   
    @unpack α, β, δ = para
    A_ss = 1.0
    k_ss = ((1/β - (1-δ)) / (α * A_ss))^(1/(α-1))
    y_ss = A_ss * k_ss^α
    c_ss = y_ss - δ * k_ss

    para.k_ss = k_ss
    para.c_ss = c_ss
    para.y_ss = y_ss
    para.A_ss = A_ss
    return k_ss, c_ss, y_ss, A_ss
end


para = Params()
println("Steady-state capital (A=1) ≈ ", round.(steady_state(para),digits=3))

#Normalize variables
function normalize(x,x_ss)
    return (x .- x_ss) ./ x_ss
end
function denormalize(x,x_ss)
    return x .* x_ss .+ x_ss
end


#Now construct the neural network 
model = Chain(
    Dense(2, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 1, softplus)  # output layer (smooth version of RELU)
)
# If GPU available, move model to GPU
if has_metal
    model = mtl(model)
end
println("Initialized neural network model: ", model)


#Now define the training functions

# Function to sample a batch of random (normalized) states (k, A)
function sample_batch(para::Params, batch_size)
    @unpack σ_ε, ρ = para
    k_ss, c_ss, y_ss, A_ss = steady_state(para)
    # sample capital around steady state (log-uniform or uniform in [0.5 k_ss, 1.5 k_ss])
    k_low, k_high = -0.5,0.5  #remember this is relative to k_ss

    # We sample uniform in log(k) to ensure lower end has coverage (since distribution of k might be skewed)
    k_batch = rand(batch_size) .* (k_high - k_low) .+ k_low
    # sample productivity A (log-normal around 1)
    # We approximate the ergodic distribution of A_t as lognormal with mean 0 and std derived from σ_ε.
    # To be precise, if A follows log AR(1), its stationary distribution is N(0, σ_stat^2) with σ_stat^2 = σ_ε^2 / (1-ρ^2).
    σ_stat = σ_ε / sqrt(1-ρ^2)
    A_batch = exp.(σ_stat * randn(batch_size)) .- A_ss

    return Float32.(k_batch), Float32.(A_batch)
end


# Euler residual for a batch (vectorized)
function euler_residual_batch(para,model, k_batch, A_batch)
    @unpack α, β, δ, γ, σ_ε, ρ, k_ss, c_ss, y_ss, A_ss = para
    batch_size = length(k_batch)
    k_low = 0.5f0*k_ss
    k_high = 1.5f0*k_ss
    # compute consumption from network
    # Prepare input matrix for model: 2 x N (each column is [k; A])
    # If on GPU, ensure k_batch, A_batch are CuArrays
    if has_cuda
        k_batch = CuArray(k_batch)
        A_batch = CuArray(A_batch)
    end
    if has_metal
        k_batch = mtl(k_batch)
        A_batch = mtl(A_batch)
    end
    X = hcat(k_batch, A_batch)'  # shape 2 x batch_size
    # Network output (on GPU or CPU depending on model)
    # model(X) will give a 1 x N matrix (or vector) of outputs
    net_out = model(X)            # 1 x batch_size

    frac = vec(net_out)        # reshape to 1D vector length = batch_size
    #aiming to output fraction
    k = denormalize(k_batch,k_ss)
    A = denormalize(A_batch,A_ss)
    # Compute consumption = frac * [A*k^α + (1-δ)k] for each state
    # Use broadcasting to compute vectorized
    c = frac .* (A .* (k .^ α) .+ (1f0 .- δ).*k)

    # Now compute Euler residual for each element
    # u'(c) = c^(-γ)
    #marginal_utility = c .^ (-γ)
    # Next state capital k' for each state:
    k_next = (1f0 .- δ).*k .+ A .* (k .^ α) .- c
    k_next = clamp.(k_next, k_low, k_high)
    c  = (1f0 .- δ).*k .+ A .* (k .^ α) .- k_next
    marginal_utility = c .^ (-γ)
 
    # Draw one sample for next shock A' (for each current state) to approximate expectation
    # We draw A' given current A: log A' = ρ log A + σε * ε. We sample ε ~ N(0,1).
    eps = randn(Float32,batch_size)
    # If GPU, convert eps to CuArray as well
    if has_cuda
        eps = CuArray(eps)
    end
    if has_metal
        eps = mtl(eps)
    end
    A_next = exp.(log.(A).*ρ .+ σ_ε .* eps)
    A_next_norm = normalize(A_next,A_ss)
    k_next_norm = normalize(k_next,k_ss)
    # Compute next-period consumption by feeding (k_next, A_next) through policy network
    X_next = hcat(k_next_norm, A_next_norm)'
    frac_next = vec(model(X_next))

    resource_next = A_next .* (k_next .^ α) .+ (1f0 .- δ).*k_next
    k_next_next = (1f0 .-frac_next).*resource_next
    k_next_next = clamp.(k_next_next, k_low, k_high)
    c_next = resource_next - k_next_next
    #c_next = (1f0 .- δ).*k_next .+ A_next .* (k_next .^ α) .- k_next
    # marginal utility next period
    marginal_utility_next = c_next .^ (-γ)

    # Compute RHS of Euler: β * u'(c_{t+1}) * (α A' k'^{α-1} + 1-δ)
    # Calculate return factor (gross) = α A' k'^{α-1} + 1 - δ
    R_next = α .* A_next .* (k_next .^ (α .- 1f0)) .+ (1f0 .- δ)
    rhs = β .* marginal_utility_next .* R_next

    # Residual = LHS - RHS
    residuals = rhs .- marginal_utility
    return residuals  # vector of length batch_size
end


# Hyperparameters
batch_size = 2048      # number of state points per batch
epochs     = 50000      # training epochs
η          = 1e-3      # initial learning rate

# Initialize optimizer
opt_state = Flux.setup(Adam(η), model)
paragpu = GPUParams(para,batch_size)

# Define loss function
function loss_fn(model,data)
    k_batch, A_batch = data
    residuals = euler_residual_batch(para, model, k_batch, A_batch)
    return mean(residuals .^ 2f0)
end



# Training loop
println("Starting training...")
losses = []
for epoch in 1:epochs
    # Sample a batch of states
    data = sample_batch(para, batch_size)
    # Compute gradient of loss (mean squared residual) w.r.t. model parameters
    val, grads = Flux.withgradient(model) do m
        loss_fn(m, data)
    end
    Flux.update!(opt_state, model, grads[1])
    # (Optional) decay learning rate or print progress
    push!(losses, val)
    if epoch % 100 == 0
        # Compute current loss for reporting
        println("Epoch $epoch, training MSE Euler residual = $(round(val, sigdigits=3))")
    end
end

# Function to simulate the economy using the trained model
function simulate_economy(model, para::Params, T::Int=200, k0=nothing, A0=nothing)
    @unpack α, β, δ, γ, ρ, σ_ε, k_ss, c_ss, y_ss, A_ss = para
    k_low = 0.5f0*k_ss
    k_high = 1.5f0*k_ss
    # If initial values not provided, use steady state values
    if isnothing(k0)
        k0 = k_ss
    end
    if isnothing(A0)
        A0 = A_ss
    end
    
    # Initialize arrays to store time series
    k_series = zeros(Float32, T+1)
    A_series = zeros(Float32, T+1)
    c_series = zeros(Float32, T)
    y_series = zeros(Float32, T)
    i_series = zeros(Float32, T)
    
    # Set initial values
    k_series[1] = k0
    A_series[1] = A0
    
    # Generate all random shocks in advance
    ε_series = randn(Float32, T)
    
    # Simulation loop
    for t in 1:T
        # Current state
        k = k_series[t]
        A = A_series[t]
        
        # Normalize state for model input
        k_norm = normalize(k, k_ss)
        A_norm = normalize(A, A_ss)
        
        # Create input for neural network
        state = reshape([k_norm, A_norm], 2, 1)
        
        # Get consumption fraction from model
        frac = model(state)[1]
        
        # Calculate current output
        y = A * k^α
        y_series[t] = y
        
        # Calculate consumption
        resources = y + (1f0 - δ) * k
        c = frac * resources
        c_series[t] = c
        
        # Calculate investment
        i = y - c
        i_series[t] = i
        
        # Next-period capital - FIX: proper capital accumulation equation
        k_next = (1f0 - δ) * k + i  # This was incorrect before (was just i)
        k_series[t+1] = max(k_next, 1e-6)  # Prevent negative capital
        
        # Next-period productivity (AR(1) process)
        logA_next = ρ * log(A) + σ_ε * ε_series[t]
        A_series[t+1] = exp(logA_next)
    end
    
    # Create results dictionary
    results = Dict(
        "capital" => k_series,
        "productivity" => A_series,
        "consumption" => c_series,
        "output" => y_series,
        "investment" => i_series,
        "savings_rate" => i_series ./ y_series
    )
    
    return results
end

# Function to analyze and plot simulation results
function analyze_simulation(results, para::Params)
    @unpack k_ss, c_ss, y_ss, A_ss = para
    
    # Calculate statistics (excluding first 50 periods as burn-in)
    burn_in = 50
    T = length(results["consumption"])
    
    # Mean values
    mean_k = mean(results["capital"][burn_in+1:end])
    mean_c = mean(results["consumption"][burn_in+1:end])
    mean_y = mean(results["output"][burn_in+1:end])
    mean_i = mean(results["investment"][burn_in+1:end])
    
    # Standard deviations
    std_k = std(results["capital"][burn_in+1:end])
    std_c = std(results["consumption"][burn_in+1:end])
    std_y = std(results["output"][burn_in+1:end])
    std_i = std(results["investment"][burn_in+1:end])
    
    # Print summary statistics
    println("Simulation Results (excluding burn-in of $burn_in periods):")
    println("Mean Capital: $mean_k (rel. to steady state: $(mean_k/k_ss))")
    println("Mean Consumption: $mean_c (rel. to steady state: $(mean_c/c_ss))")
    println("Mean Output: $mean_y (rel. to steady state: $(mean_y/y_ss))")
    println("Mean Investment: $mean_i (rel. to steady state: $(mean_i/(y_ss-c_ss)))")
    println("Mean Savings Rate: $(mean_i/mean_y)")
    println("\nStandard Deviations:")
    println("σ(k)/k̄ = $(std_k/mean_k)")
    println("σ(c)/c̄ = $(std_c/mean_c)")
    println("σ(y)/ȳ = $(std_y/mean_y)")
    println("σ(i)/ī = $(std_i/mean_i)")
    
    
    # Time series plots
    p1 = plot(0:T, results["capital"], label="Capital", title="Capital", linewidth=2)
    hline!([k_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    p2 = plot(1:T, results["consumption"], label="Consumption", title="Consumption", linewidth=2)
    hline!([c_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    p3 = plot(1:T, results["productivity"], label="Productivity", title="Productivity Shock", linewidth=2)
    hline!([A_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    p4 = plot(1:T, results["investment"], label="Investment", title="Investment", linewidth=2)
    hline!([y_ss-c_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
    # Return both statistics and plot
    return combined_plot
end

# Function to compute impulse response to productivity shock
function impulse_response(model, para::Params, shock_size=0.05, T=40)
    # Run two simulations: one baseline and one with a shock
    # Initial conditions: steady state
    Random.seed!(96)
    baseline = simulate_economy(model, para, T, para.k_ss, para.A_ss)
    
    # Shocked economy: shock_size% higher productivity initially
    Random.seed!(96)
    shocked = simulate_economy(model, para, T, para.k_ss, para.A_ss * (1 + shock_size))
    
    # Calculate percentage deviations from baseline
    deviations = Dict{String, Vector{Float32}}()
    for key in ["capital", "consumption", "output", "investment"]
        if key == "capital"
            # Capital has T+1 elements
            deviations[key] = 100 * (shocked[key][1:T] ./ baseline[key][1:T] .- 1)
        else
            deviations[key] = 100 * (shocked[key] ./ baseline[key] .- 1)
        end
    end
    
    # Plot impulse responses
    p = plot(title="Impulse Responses to $(shock_size*100)% Productivity Shock", 
             xlabel="Time", ylabel="% Deviation from Baseline",
             legend=:topright, size=(800, 500))
    
    plot!(p, 1:T, deviations["output"], label="Output", linewidth=2)
    plot!(p, 1:T, deviations["consumption"], label="Consumption", linewidth=2)
    plot!(p, 1:T, deviations["investment"], label="Investment", linewidth=2)
    plot!(p, 1:T, deviations["capital"], label="Capital", linewidth=2)
    
    return p
end

# Run a simulation after training
println("Running economic simulation with trained model...")
sim_results = simulate_economy(model, para, 500)
plot = analyze_simulation(sim_results, para)
display(plot)

# Generate impulse response
println("\nComputing impulse response to productivity shock...")
ir_plot = impulse_response(model, para)
display(ir_plot)

# Policy function visualization
function plot_policy_function(model, para::Params)
    @unpack k_ss, A_ss = para
    
    # Create grid of capital and productivity values
    k_grid = range(0.5 * k_ss, 1.5 * k_ss, length=50)
    A_values = [0.9 * A_ss, A_ss, 1.1 * A_ss]  # Low, medium, high productivity
    
    # Compute consumption for each state
    results = Dict()
    for A in A_values
        c_values = Float32[]
        for k in k_grid
            # Normalize inputs
            k_norm = normalize(k, k_ss)
            A_norm = normalize(A, A_ss)
            
            # Get policy
            frac = model(reshape([k_norm, A_norm], 2, 1))[1]
            
            # Compute consumption
            resources = A * k^para.α + (1f0 - para.δ) * k
            c = frac * resources
            push!(c_values, c)
        end
        results[A] = c_values
    end
    
    # Plot policy functions
    p = plot(title="Consumption Policy Functions", 
             xlabel="Capital", ylabel="Consumption",
             legend=:bottomright, size=(800, 500))
    
    plot!(p, k_grid, results[0.9 * A_ss], label="Low Productivity (A=0.9)", linewidth=2)
    plot!(p, k_grid, results[A_ss], label="Normal Productivity (A=1.0)", linewidth=2)
    plot!(p, k_grid, results[1.1 * A_ss], label="High Productivity (A=1.1)", linewidth=2)
    
    return p
end

# Visualize the policy function
policy_plot = plot_policy_function(model, para)
display(policy_plot)



