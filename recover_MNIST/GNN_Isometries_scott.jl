using BSON: @load
using Flux
using Flux: params
using Flux.Optimise
using Flux.Optimise: update!
using Flux.Data: DataLoader
using ImageFiltering
using Images
using ImageIO
using MLDatasets: FashionMNIST
using LinearAlgebra
using MLDatasets
using Plots
using Zygote
using FFTW
using Distributions
using Statistics
using SparseArrays
using JLD
using Hyperopt
using Plots
using LaTeXStrings
##

"""
    optimise(init_z, loss, opt, tolerance, [out_toggle = 0,][max_iter = 1_000_000])

    Optimization that stops when the gradient is small enough
"""
##

function sample_fourier(aimed_m, n)
    F = dct(diagm(ones(n)),2)
    sampling = rand(Bernoulli(aimed_m/n), n)
    true_m = sum(sampling)
    return (true_m, F[sampling,:]* sqrt(n/true_m)) # normalize it
end
function fourier(dim)
    dct(diagm(ones(dim)),2)
end
##

##
function subspace_incoherence(F::Matrix{Number}, A::Matrix{Number})
    α = 0
    m, _ = size(A)
    QR_decomp = qr(A)
    for i in 1:m
        temp = norm(Matrix(QR_decomp.Q)'*F[i,:], 2)
        α = max(α, temp)
    end
    return α
end


##

include("./SignalRecovery.jl")
include("./Plotting_Functions.jl")

##
function getVAE(;encoder_μ, encoder_logvar, W1::Matrix{T}, W2::Matrix{T}, W3::Matrix{T}, Q::Matrix{T}) where T <: Real
    function(x::AbstractArray{T})
        μ = encoder_μ(x)
        logvar = encoder_logvar(x)
        # Apply reparameterisation trick to sample latent
        z = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)
        # Reconstruct from latent sample
        Q*W3*relu(W2*relu(W1*z))
    end
end
##
function getVAEgenerator(;W1::Matrix{T}, W2::Matrix{T}, W3::Matrix{T}, Q::Matrix{T}, with_sigmoid=false, kwargs...) where T <: Real
    if with_sigmoid
        return Chain(Dense(W1,false,relu), Dense(W2, false, relu), Dense(W3, false, identity), Dense(Q, false, sigmoid))
    else
        return Chain(Dense(W1,false,relu), Dense(W2, false, relu), Dense(W3, false, identity), Dense(Q, false, identity))
    end
end

##
weights = load("./trained_GNN/MNIST_identity_v4/model-40.bson")
weights[:W3] = convert.(Float64, weights[:W3])
fullVAE = getVAE(;weights...)
generatorVAE = getVAEgenerator(;weights...)
dataset = MNIST(Float64, split=:test)
##


# Experiment to Recover data directly on the MNIST image with maladapted Generator.
index = 39 #index of the image in the dataset
meas = [10 , 15, 20, 25, 50, 100, 250, 500]
recoveries = Array{Any, 1}(undef, length(meas))
for (i,m) in enumerate(meas)
    recoveries[i] = recover_signal(reshape(1 .- dataset[index].features, 28^2), 20, sample_fourier(m,28^2)[2], generatorVAE, tolerance = 1e-4, max_iter=5000)
end
recoveries
plotmeasurementMNISTarray(meas=meas, recoveries=recoveries, true_signal = dataset[index].features)
#colorview(Gray, reshape(generatorVAE(randn(20)/sqrt(20)), 28,28)')
#plot the recoveries.

##

# Experiment: recover before the sigmoid
# This seems to fail because of a bad metric on the images
function inversesigmoid(y)
    y = y < (1 - 1e-6) ? y : 1-1e-10
    y = y > 1e-6 ? y : 1e-10
    log(y/(1-y))
end
##
generatorVAE = getVAEgenerator(with_sigmoid=false; weights...)
##
index = 3
m = 600
#signal = inversesigmoid.(reshape(1 .- dataset[index].features, 28^2))
signal = inverse_sigmoid.(reshape(1 .- dataset[index].features, 28^2))
colorview(Gray, 1 .- dataset[index].features')
##
recovery = recover_signal(signal, 20, sample_fourier(m, 28^2)[2], generatorVAE, max_iter = 7_000)
colorview(Gray, sigmoid.(reshape(recovery, 28, 28))')
##
# β =0.6
# k=20
# mid = 500
# n = 500
# desired_m = 24
# F = fourier(n)
# m, A = sample_fourier(desired_m, n)
# model = Chain(
#         Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
#         Dense(mid, n, bias = false; init =(out,in) -> randn(out,in)/sqrt(out))
#     )
# W₁ = Flux.params(model)[1]
# W₂ = Flux.params(model)[2]
# opt = Flux.Optimise.Momentum(0.0005)
# out_log = 5_000
# num_data = 5
# tolerance = 1e-6
# results = [[],[]]
# max_iter = 12_000
##
#inco = subspace_incoherence(F,W₂)
#W₁ = randn(mid,k)/sqrt(mid)
#W₂ = β*randn(n,mid)/sqrt(n) + (1-β) * F[:,1:mid]

# model = Chain(
#         Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
#         Dense(mid, n, bias = false; init =(out,in) -> β*randn(out,in)/sqrt(out) + (1-β) * F[:,1:in])
#     )

# ho 
# ##
# ho = @hyperopt for i = 300,
#         tolerance = exp10.(LinRange(-12,0,30)),
#         k = [20,40],
#         lr = exp10.(LinRange(-9,2,40)),
#         max_iter = 10_000,
#         n = [200,500]

#     cost = solo_recovery_error(k , n , n, 2*k, 0.5, lr, tolerance, max_iter)
# end
# printmin(ho)
# plot(ho)

##





##
# scatter(results[1], results[2])

# results: tolerance 10^-7, lr=0.1

# plot_by_meas = [(m, solo_recovery_error(20, 200, 200, m, 0.5, 0.1,1e-7, 10_000)) for m in 5:5:100]
# plot(plot_by_meas)
##


##
# for β in 0:0.1:1
#     inco = subspace_incoherence(F,W₂)
#     W₁ .= randn(mid,k)
#     W₂ .= β*randn(n,mid)/sqrt(n) + (1-β) * F[:,1:mid]
#     for i in 1:num_data
#         push!(results[1], inco)
#         push!(results[2], recovery_error(k,A, model, opt, tolerance = tolerance) )
#     end
# end
##
# function solo_recovery_error(k, mid, n::Integer, desired_m::Integer ,β::Float64 , lr::Float64=0.1, tolerance::Float64=1e-7 , max_iter::Integer = 10_000)
#     F = fourier(n)
#     model = Chain(
#         Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
#         Dense(mid, n, bias = false; init =(out,in) -> β*randn(out,in)/sqrt(out) + (1-β) * F[:,1:in])
#     )
#     m, A = sample_fourier(desired_m, n)
#     out_log = 5000
#     opt = Flux.Optimise.ADAM(lr)
#     recovery_error(k,A, model, opt, tolerance = tolerance, max_iter = max_iter)
# end
##


##
function run_recovery_random_weights_experiment()
    k=16
    mid = 32
    n = 64
    desired_m = 24
    F = fourier(n)
    m, A = sample_fourier(desired_m, n)

    W₁ = Flux.params(model)[1]
    W₂ = Flux.params(model)[2]
    opt = Flux.Optimise.Momentum(5e-5)
    num_data = 12
    tolerance = 2e-6
    results = [[],[]]
    model = Chain(
        Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
        Dense(mid, n, bias = false; init =(out,in) -> β*randn(out,in)/sqrt(out) + (1-β) * F[:,1:in])
    )
    for β in 0.0:0.02:1
        W₁ .= randn(mid,k)
        W₂ .= β*randn(n,mid)/sqrt(n) + (1-β) * F[:,1:mid]

        inco = subspace_incoherence(F,params(model)[2])
        for i in 1:num_data
            push!(results[1], inco)
            push!(results[2], recovery_error(k,A, model, opt, tolerance = tolerance) )
            println("pushed")
        end
    end
    return results
end
##
#results = run_experiment()
##
#@save "./results_6_tolerance.jld" results
##

##

#plot_scatter_stats(results)