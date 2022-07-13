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

function optimise(opt, z, loss; tolerance, out_toggle, max_iter = 500_000)
    tol2 = tolerance^2
    ps = params(z)
    iter=1
    println("starting loss: ", loss())
    while true
        if iter > max_iter
            @warn "Max num. iterations reached"
            return nothing
        end
        grads = gradient(loss, ps) #loss cannot have any arguments
        update!(opt, ps, grads)
        succ_error = sum(abs2, grads[z])
        if out_toggle != 0 && iter % out_toggle == 0
            println("====> In Gradient: Iteration: $iter grad_size: $(sqrt(succ_error)) tolerance: $tolerance  loss: ", string(loss()))
        end
        if succ_error < tol2
            break
        end
        iter += 1
    end
    return z
end


##

function subspace_incoherence(F, A)
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
function recovery_error(k, A, model, opt; tolerance=1e-6, max_iter = 12_000)
    x₀ = model(randn(k)/sqrt(k))
    k̂ = randn(k)/sqrt(k)
    y= A*x₀
    function loss()
        return sum(abs2, A*model(k̂) - y)
    end
    optimise(opt, k̂, loss, tolerance=tolerance, out_toggle = 1e5, max_iter = max_iter)
    return norm(model(k̂) - x₀)
end
##
β =0.6
k=20
mid = 500
n = 500
desired_m = 24
F = fourier(n)
m, A = sample_fourier(desired_m, n)
model = Chain(
        Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
        Dense(mid, n, bias = false; init =(out,in) -> randn(out,in)/sqrt(out))
    )
W₁ = Flux.params(model)[1]
W₂ = Flux.params(model)[2]
opt = Flux.Optimise.Momentum(0.0005)
out_log = 5_000
num_data = 5
tolerance = 1e-6
results = [[],[]]
max_iter = 12_000
##
#inco = subspace_incoherence(F,W₂)
#W₁ = randn(mid,k)/sqrt(mid)
#W₂ = β*randn(n,mid)/sqrt(n) + (1-β) * F[:,1:mid]

model = Chain(
        Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
        Dense(mid, n, bias = false; init =(out,in) -> β*randn(out,in)/sqrt(out) + (1-β) * F[:,1:in])
    )

ho 
##
ho = @hyperopt for i = 300,
        tolerance = exp10.(LinRange(-12,0,30)),
        k = [20,40],
        lr = exp10.(LinRange(-9,2,40)),
        max_iter = 10_000,
        n = [200,500]

    cost = solo_recovery_error(k , n , n, 2*k, 0.5, lr, tolerance, max_iter)
end
printmin(ho)
plot(ho)

##
logscale(params::AbstractVector{T}) where T <: Real = /(extrema(params)...) < 2e-2
@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    for i = 1:N
        params = getindex.(ho.history, i)
        if eltype(params) <: Real 
            perm = sortperm(params) 
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> "Sampled points"
                legend --> false
                xscale --> (logscale(params) ? :log10 : :identity)
                yscale --> (logscale(ho.results) ? :log10 : :identity)
                params[perm], ho.results[perm]
            end
        else
            params = Symbol.(params)
            uniqueparams = sort(unique(params))
            paramidxs = map(x -> findfirst(==(x), uniqueparams), params)
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> "Sampled points"
                legend --> false
                xticks --> (1:length(uniqueparams), uniqueparams)
                xscale --> :identity
                yscale --> (logscale(ho.results) ? :log10 : :identity)
                paramidxs, ho.results
            end
        end
    end
end




##
scatter(results[1], results[2])

# results: tolerance 10^-7, lr=0.1

plot_by_meas = [(m, solo_recovery_error(20, 200, 200, m, 0.5, 0.1,1e-7, 10_000)) for m in 5:5:100]
plot(plot_by_meas)
##


##
for β in 0:0.1:1
    inco = subspace_incoherence(F,W₂)
    W₁ .= randn(mid,k)
    W₂ .= β*randn(n,mid)/sqrt(n) + (1-β) * F[:,1:mid]
    for i in 1:num_data
        push!(results[1], inco)
        push!(results[2], recovery_error(k,A, model, opt, tolerance = tolerance) )
    end
end
##
function solo_recovery_error(k, mid, n::Integer, desired_m::Integer ,β::Float64 , lr::Float64=0.1, tolerance::Float64=1e-7 , max_iter::Integer = 10_000)
    F = fourier(n)
    model = Chain(
        Dense(k, mid, relu, bias = false; init =(out,in) -> randn(out,in)/sqrt(out)),
        Dense(mid, n, bias = false; init =(out,in) -> β*randn(out,in)/sqrt(out) + (1-β) * F[:,1:in])
    )
    m, A = sample_fourier(desired_m, n)
    out_log = 5000
    opt = Flux.Optimise.ADAM(lr)
    recovery_error(k,A, model, opt, tolerance = tolerance, max_iter = max_iter)
end
##


##
function run_experiment()
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
results = run_experiment()
##
@save "./results_6_tolerance.jld" results
##

function plot_scatter_stats(results:: AbstractArray{Tuple{Real,2}})
    incoherences = collect(Set(results[1]))
    means = [mean(results[2][i] for i in 1 : length(results[1]) if results[1][i] == incoherence) for incoherence in incoherences]
    deviations = [std( results[2][i] for i in 1 : length(results[1]) if results[1][i] == incoherence) for incoherence in incoherences]

    stats = cat(incoherences,means, deviations, dims=2)
    sorted_stats = sortslices(stats, dims=1)
    scatter(results[1], results[2])
    plot!(sorted_stats[:,1], sorted_stats[:,2], linewidth = 3.2, ribbon= sorted_stats[:,3],fillalpha=.5)
end
##

plot_scatter_stats(results)