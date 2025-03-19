### Julia environment setup
using Flux
using Statistics
using Random
using Parameters
using Metal
using Plots
using DataFrames
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


    #bounds
    k_bounds::Tuple{Float32,Float32} = (0.5f0,1.5f0)
    A_bounds::Tuple{Float32,Float32} = (0.5f0,1.5f0)
    β_bounds::Tuple{Float32,Float32} = (0.9f0,0.99f0)
    ρ_bounds::Tuple{Float32,Float32} = (0.9f0,0.99f0)
end

### Utility function and its marginal derivative
function u(c,γ)
    return c^(1-γ) / (1-γ)
end
function u_prime(c,γ)
    return c.^(-γ)      # derivative of c^(1-γ)/(1-γ) is c^(-γ)
end

# Steady-state (for A=1) for reference (solve α β (A) k^(α-1) + β(1-δ) = 1)
function steady_state(α, β, δ)   
    @unpack α, β, δ = para
    A_ss = 1.0f0
    k_ss = @. ((1f0/β - (1f0-δ)) / (α * A_ss))^(1f0/(α-1f0))
    y_ss = @. A_ss * k_ss^α
    c_ss = @. y_ss - δ * k_ss

    return k_ss, c_ss, y_ss, A_ss
end


para = Params()
k_ss, c_ss, y_ss, A_ss = steady_state(para.α, para.β, para.δ)
println("Steady-state capital (A=1) ≈ ", round.(k_ss,digits=3))

#Normalize variables
function normalize(x,x_low,x_high)
    return (x .- x_low) ./ (x_high - x_low)
end
function denormalize(x,x_low,x_high)
    return x .* (x_high - x_low) .+ x_low
end


#Now construct the neural network 
model = Chain(
    Dense(4, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1, softplus)  # output layer (smooth version of RELU)
)
# If GPU available, move model to GPU
if has_metal
    model = mtl(model)
end
println("Initialized neural network model: ", model)


#Now define the training functions

# Function to sample a batch of random (normalized) states (k, A)
function sample_batch(para::Params, batch_size)
    @unpack σ_ε, ρ, k_bounds, A_bounds, β_bounds = para
    # We sample uniform in log(k) to ensure lower end has coverage (since distribution of k might be skewed)
    k_batch = rand(Float32,batch_size)
    β_batch = rand(Float32,batch_size)
    ρ_batch = rand(Float32,batch_size)
    # sample productivity A (log-normal around 1)
    # We approximate the ergodic distribution of A_t as lognormal with mean 0 and std derived from σ_ε.
    # To be precise, if A follows log AR(1), its stationary distribution is N(0, σ_stat^2) with σ_stat^2 = σ_ε^2 / (1-ρ^2).
    σ_stat = σ_ε ./ sqrt.(1f0 .- ρ.^2f0)
    A = exp.(σ_stat .* randn(Float32,batch_size))
    A_batch = normalize(A,A_bounds[1],A_bounds[2])

    return Float32.(k_batch), Float32.(A_batch), Float32.(β_batch), Float32.(ρ_batch)
end


# Euler residual for a batch (vectorized)
function euler_residual_batch(para,model, data)
    @unpack α, β, δ, γ, σ_ε, ρ, k_bounds, A_bounds,β_bounds,ρ_bounds = para
    # compute consumption from network
    # Prepare input matrix for model: 2 x N (each column is [k; A])
    # If on GPU, ensure k_batch, A_batch are CuArrays
    k_batch, A_batch, β_batch, ρ_batch = data
    if has_cuda
        k_batch = CuArray(k_batch)
        A_batch = CuArray(A_batch)
        β_batch = CuArray(β_batch)
        ρ_batch = CuArray(ρ_batch)
    end
    if has_metal
        k_batch = mtl(k_batch)
        A_batch = mtl(A_batch)
        β_batch = mtl(β_batch)
        ρ_batch = mtl(ρ_batch)
    end

    β = denormalize(β_batch,β_bounds[1],β_bounds[2])
    ρ = denormalize(ρ_batch,ρ_bounds[1],ρ_bounds[2])
    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)
    batch_size = length(k_batch)
    k_low = k_bounds[1]*k_ss
    k_high = k_bounds[2]*k_ss
    
    X = hcat(k_batch, A_batch, β_batch, ρ_batch)'  # shape 4 x batch_size
    # Network output (on GPU or CPU depending on model)
    # model(X) will give a 1 x N matrix (or vector) of outputs
    net_out = model(X)            # 1 x batch_size

    frac = vec(net_out)        # reshape to 1D vector length = batch_size
    #aiming to output fraction
    k = denormalize(k_batch,k_low,k_high)
    A = denormalize(A_batch,A_bounds[1],A_bounds[2])
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
    A_next_norm = normalize(A_next,A_bounds[1],A_bounds[2])
    k_next_norm = normalize(k_next,k_low,k_high)
    # Compute next-period consumption by feeding (k_next, A_next) through policy network
    X_next = hcat(k_next_norm, A_next_norm, β_batch, ρ_batch)'
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

# Define loss function
function loss_fn(model,data)
    residuals = euler_residual_batch(para, model, data)
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
    @unpack α, β, δ, γ, ρ, σ_ε, k_bounds, A_bounds,β_bounds,ρ_bounds = para
    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)
    k_low = k_bounds[1]*k_ss
    k_high = k_bounds[2]*k_ss
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
        k_norm = normalize(k, k_low, k_high)
        A_norm = normalize(A, A_bounds[1], A_bounds[2])
        β_norm = normalize(β, β_bounds[1], β_bounds[2])
        ρ_norm = normalize(ρ, ρ_bounds[1], ρ_bounds[2])
        
        # Create input for neural network
        state = reshape([k_norm, A_norm, β_norm, ρ_norm], 4, 1)
        
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
    
    # Create DataFrame with results
    results = DataFrame(
        capital = k_series[1:T],
        productivity = A_series[1:T],
        consumption = c_series,
        output = y_series,
        investment = i_series,
        savings_rate = i_series ./ y_series
    )
    
    # Add metadata as properties
    results.k_ss = k_ss*ones(T)
    results.c_ss = c_ss*ones(T)
    results.y_ss = y_ss*ones(T)
    results.A_ss = A_ss*ones(T)
    results.β = β*ones(T)
    results.ρ = ρ*ones(T)
    
    return results
end

# Function to analyze and plot simulation results
function analyze_simulation(results::DataFrame, para::Params)
    # Extract steady state values from results metadata
    k_ss = results.k_ss[1]
    c_ss = results.c_ss[1]
    y_ss = results.y_ss[1]
    A_ss = results.A_ss[1]
    
    # Extract parameters used in simulation
    β_sim = results.β[1]
    ρ_sim = results.ρ[1]
    
    # Calculate statistics (excluding first 50 periods as burn-in)
    burn_in = 50
    T = size(results, 1)
    
    # Mean values
    mean_k = mean(results.capital[burn_in+1:end])
    mean_c = mean(results.consumption[burn_in+1:end])
    mean_y = mean(results.output[burn_in+1:end])
    mean_i = mean(results.investment[burn_in+1:end])
    
    # Standard deviations
    std_k = std(results.capital[burn_in+1:end])
    std_c = std(results.consumption[burn_in+1:end])
    std_y = std(results.output[burn_in+1:end])
    std_i = std(results.investment[burn_in+1:end])
    
    # Print summary statistics
    println("Simulation Results with β = $β_sim, ρ = $ρ_sim (excluding burn-in of $burn_in periods):")
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
    p1 = plot(0:T-1, results.capital, label="Capital", title="Capital", linewidth=2)
    hline!([k_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    p2 = plot(0:T-1, results.consumption, label="Consumption", title="Consumption", linewidth=2)
    hline!([c_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    p3 = plot(0:T-1, results.productivity, label="Productivity", title="Productivity Shock", linewidth=2)
    hline!([A_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    p4 = plot(0:T-1, results.investment, label="Investment", title="Investment", linewidth=2)
    hline!([y_ss-c_ss], linestyle=:dash, color=:red, label="Steady State", alpha=0.7)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800),
                        plot_title="Simulation with β=$(round(β_sim,digits=3)), ρ=$(round(ρ_sim,digits=3))")
    
    # Return both statistics and plot
    return combined_plot
end

# Function to compute impulse response to productivity shock
function impulse_response(model, para::Params, shock_size=0.05, T=40,NIRF=20)
    k_ss, c_ss, y_ss, A_ss = steady_state(para.α, para.β, para.δ)
    # Run two simulations: one baseline and one with a shock
    # Initial conditions: steady state
    Random.seed!(96)
    baseline = simulate_economy(model, para, T, k_ss, A_ss)
    for i in 1:NIRF-1
        baseline = baseline .+ simulate_economy(model, para, T, k_ss, A_ss)
    end
    baseline = baseline ./ NIRF
    # Shocked economy: shock_size% higher productivity initially
    Random.seed!(96)
    shocked = simulate_economy(model, para, T, k_ss, A_ss * (1 + shock_size))
    for i in 1:NIRF-1
        shocked = shocked .+ simulate_economy(model, para, T, k_ss, A_ss * (1 + shock_size))
    end
    shocked = shocked ./ NIRF
    
    # Calculate percentage deviations from baseline
    deviations = DataFrame()
    
    # Calculate deviations for each variable
    deviations.output = 100 * (shocked.output ./ baseline.output .- 1)
    deviations.consumption = 100 * (shocked.consumption ./ baseline.consumption .- 1)
    deviations.investment = 100 * (shocked.investment ./ baseline.investment .- 1)
    deviations.capital = 100 * (shocked.capital ./ baseline.capital .- 1)
    
    # Add metadata
    deviations.β = baseline.β
    deviations.ρ = baseline.ρ
    
    # Plot impulse responses
    p = plot(title="Impulse Responses to $(shock_size*100)% Productivity Shock (β=$(round(baseline.β[1],digits=3)), ρ=$(round(baseline.ρ[1],digits=3)))", 
             xlabel="Time", ylabel="% Deviation from Baseline",
             legend=:topright, size=(800, 500))
    
    plot!(p, 0:T-1, deviations.output, label="Output", linewidth=2)
    plot!(p, 0:T-1, deviations.consumption, label="Consumption", linewidth=2)
    plot!(p, 0:T-1, deviations.investment, label="Investment", linewidth=2)
    plot!(p, 0:T-1, deviations.capital, label="Capital", linewidth=2)
    
    return p, deviations
end

# Run a simulation after training
println("Running economic simulation with trained model...")
sim_results = simulate_economy(model, para, 500)
sim_plot = analyze_simulation(sim_results, para)
display(sim_plot)

# Generate impulse response
println("\nComputing impulse response to productivity shock...")
ir_plot, deviations = impulse_response(model, para, 0.05, 40, 100)
display(ir_plot)

# Policy function visualization
function plot_policy_function(model, para::Params)
    @unpack k_bounds, A_bounds, α, δ, β, β_bounds, ρ, ρ_bounds = para
    k_ss, c_ss, y_ss, A_ss = steady_state(α, β, δ)
    k_low = k_bounds[1]*k_ss
    k_high = k_bounds[2]*k_ss
    
    # Create grid of capital and productivity values
    k_grid = range(k_low, k_high, length=50)
    A_values = [0.9 * A_ss, A_ss, 1.1 * A_ss]  # Low, medium, high productivity
    
    # Create DataFrame to store policy function data
    policy_data = DataFrame()
    
    # Normalize parameter values once
    β_norm = normalize(β, β_bounds[1], β_bounds[2])
    ρ_norm = normalize(ρ, ρ_bounds[1], ρ_bounds[2])
    
    # Compute consumption for each state
    for A_label in ["Low (A=0.9)", "Normal (A=1.0)", "High (A=1.1)"]
        A_value = A_values[findfirst(x -> x == A_label, ["Low (A=0.9)", "Normal (A=1.0)", "High (A=1.1)"])]
        
        capital_values = Float32[]
        consumption_values = Float32[]
        productivity_labels = String[]
        
        for k in k_grid
            # Normalize inputs
            k_norm = normalize(k, k_low, k_high)
            A_norm = normalize(A_value, A_bounds[1], A_bounds[2])
            
            # Get policy
            frac = model(reshape([k_norm, A_norm, β_norm, ρ_norm], 4, 1))[1]
            
            # Compute consumption
            resources = A_value * k^α + (1f0 - δ) * k
            c = frac * resources
            
            # Store results
            push!(capital_values, k)
            push!(consumption_values, c)
            push!(productivity_labels, A_label)
        end
        
        # Add data for this productivity level
        temp_df = DataFrame(
            capital = capital_values,
            consumption = consumption_values,
            productivity = productivity_labels
        )
        
        if isempty(policy_data)
            policy_data = temp_df
        else
            policy_data = vcat(policy_data, temp_df)
        end
    end
    
    # Plot policy functions
    p = plot(title="Consumption Policy Functions (β=$(round(β,digits=3)), ρ=$(round(ρ,digits=3))    )", 
             xlabel="Capital", ylabel="Consumption",
             legend=:bottomright, size=(800, 500))
    
    # Plot each productivity level
    for A_label in ["Low (A=0.9)", "Normal (A=1.0)", "High (A=1.1)"]
        subset = filter(row -> row.productivity == A_label, policy_data)
        plot!(p, subset.capital, subset.consumption, label=A_label, linewidth=2)
    end
    
    return p, policy_data
end

# Visualize the policy function
policy_plot, policy_data = plot_policy_function(model, para)
display(policy_plot)



