using BSON: @load
using LsqFit

include("reusefiles/VAE_recovery.jl")
include("reusefiles/vaemodels.jl")

@load "reusefiles/experiment_data/std_old_inco_recovery_curve_comparison.BSON" returndata

scatter(returndata["incoherent"]..., xrange=[0, 200])
scatter!(returndata["standard"]...)




@load "reusefiles/savedmodels/boundedcoherencematchingepoch20" model
model1 = model
#@load "reusefiles/savedmodels/standardandmatchingepoch20" model
#model2 = model

logdata = logrange(4, 28^2, 100)

plot_models_recovery_errors([("boundedincoherent", model, model.decoder)],
    logdata, 16, 28^2, savefile="reusefiles/experiment_data/bounded_inco_recovery_comparison.BSON")

@load "reusefiles/experiment_data/bounded_inco_recovery_comparison.BSON" returndata

@info returndata
#recoverythreshold_fromrandomimage(model, model.decoder, logdata, 16, 28^2, savefile="reusefiles/experiment_data/scatter_recoveries_v2.BSON")

train_incoherentVAE_onMNIST(vaelossfn=VAEloss_boundedcoherence, numepochs=20, label="boundedcoherencematching", α=1.0f2, λ=1.0f-2, logginglossterms=false)

