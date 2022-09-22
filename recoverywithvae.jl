using BSON: @load

include("reusefiles/VAE_recovery.jl")
include("reusefiles/relaxedrecovery.jl")

@load "reusefiles/savedmodels/more_incoherentepoch20" model


f = plot_MNISTrecoveries(model, logrange(2, 784, 2), [1, 2, 3, 4, 5, 6, 7])
@info f

firstplot
secondplot


plot_MNISTrecoveries_Makie(model, logrange(64, 784, 5), [2, 9], inrange=false)

@time plot_MNISTrecoveries(model, logrange(32, 784, 5), [2, 3, 4, 5, 6, 7, 8, 9], inrange=false)

#TODO: Compare with rng


f = Figure()


