using BSON: @load
using LsqFit
using TensorBoardLogger
using Plots
using Flux

include("reusefiles/compressedsensing.jl")
include("reusefiles/relaxedrecovery.jl")

@load "reusefiles/savedmodels/more_incoherentepoch20" model

rng = MersenneTwister(1234);

@time plot_MNISTrecoveries_bynumber_bymeasurementnumber_relaxed(model, logrange(64, 784, 6), [2, 3, 4, 5, 6, 7, 8, 9], intermediatelayers=[2], inrange=false)
@time plot_MNISTrecoveries_bynumber_bymeasurementnumber(model, logrange(64, 784, 6), [2, 3, 4, 5, 6, 7, 8, 9], inrange=false)

#TODO: Compare with rng