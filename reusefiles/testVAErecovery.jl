using Test
using BSON: @load

include("VAE_recovery.jl")

@load "reusefiles/savedmodels/bounded_morecoherencematchingepoch20" model

plot_MNISTrecoveries_bynumber_bymeasurementnumber(model, [4, 512], [5, 6], tolerance=1e-2)