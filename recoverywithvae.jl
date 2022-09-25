using BSON: @load
include("reusefiles/VAE_recovery.jl")
include("reusefiles/relaxedrecovery.jl")

@load "reusefiles/savedmodels/more_incoherentepoch20" model
incoherentmodel = model

@load "reusefiles/savedmodels/bounded_morecoherencematchingepoch20" model
boundedmodel = model



f = compare_models_MNISTrecoveries([incoherentmodel, boundedmodel], 128, 8, inrange=false, presigmoid=true)
f[1]
f[2]
@info f

firstplot
secondplot


plot_MNISTrecoveries_Makie(model, 53, [2, 9], inrange=false)

@time plot_MNISTrecoveries(model, logrange(32, 784, 5), [2, 3, 4, 5, 6, 7, 8, 9], inrange=false, presigmoid=false)

#TODO: Compare with rng



f = Figure()

@generated function tester(x; arg=2)
    return :(x * x)
end
tester(2)
throw(MethodError(plot_MNISTrecoveries))