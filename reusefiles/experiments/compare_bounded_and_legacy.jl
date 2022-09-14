### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 5c5b75cb-7214-4fae-a874-cf7c67b29045
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the global environment
    Pkg.activate("/Users/matthewscott/Prog/GNN_Isometries")
end

# ╔═╡ 2213478e-ed6d-4d05-98a7-35a69e843ca8
begin
    using BSON: @load
    using LsqFit
    using TensorBoardLogger
    using Plots
    using Flux
end

# ╔═╡ 37aa1672-830d-4e2a-9f2b-6f9289f0af0b
include("vaemodels.jl")

# ╔═╡ 0f8ef748-5c11-4189-b0b7-d82086004b46

@load "reusefiles/savedmodels/bounded_morecoherencematchingepoch20" model
newincomodel = model

@load "reusefiles/savedmodels/more_incoherentepoch20" model
oldincomodel = model

models = [newincomodel, oldincomodel]
modellabels = ["newincomodel", "oldincomodel"]
meas = logrange(40, 250, 3)

resultsdata = compare_models_from_thresholds(models, modellabels, meas, 2, 16, 28^2)

plot_models_recovery_errors([newincomodel, oldincomodel], ["bounded", "old school"], logrange(16, 28^2, 500), max_iter=1000, savefile="reusefiles/experiment_data/oldvsnew.BSON")

returnobj = recoverythreshold_fromrandomimage(oldincomodel, logrange(16, 200, 20))
returnobj.fitplot
returnobj.threshold