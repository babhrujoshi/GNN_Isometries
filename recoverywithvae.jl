using BSON: @load
using LsqFit
using TensorBoardLogger
using Plots
using Flux

include("reusefiles/VAE_recovery.jl")
include("reusefiles/compressedsensing.jl")
include("reusefiles/vaemodels.jl")

# @load "reusefiles/models/bounded_morecoherencematchingepoch20" model
# moreboundedmodel = model
# #@load "reusefiles/savedmodels/standardandmatchingepoch20" model
# #model2 = model

# logdata = logrange(24, 200, 30)

# plot_models_recovery_errors([("boundedmoreincoherent", moreboundedmodel, moreboundedmodel.decoder)],
#     logdata, 16, 28^2, savefile="reusefiles/experiment_data/more_bounded_inco_recovery_comparison.BSON")

# @load "reusefiles/experiment_data/std_old_inco_recovery_curve_comparison.BSON" returndata
# oldreturndata = returndata
# @load "reusefiles/experiment_data/bounded_inco_recovery_comparison.BSON" returndata
# returndata

# incodata = cat(oldreturndata["incoherent"]..., dims=2)

#scatter(incodata[:, 1], incodata[:, 2], label="incoherent", xaxis=:log, yaxis=:log)

#threshold_through_fit(incodata[:, 1], incodata[:, 2], sigmoid_x_scale=2.5f0)[3]

#meas = logrange(20, 200, 10)
#scatterfitplot, fit = recoverythreshold_fromrandomimage(model, model.decoder, meas, 16, 28^2)
#scatterfitplot


#include("reusefiles/VAE_recovery.jl")

#@load "reusefiles/savedmodels/bounded_morecoherencematchingepoch20" model
#plot_MNISTrecoveries_bynumber_bymeasurementnumber(model, model.decoder, [10, 30, 50, 80], [2, 5, 8], 16, 28^2, inrange=false, presigmoid=false)




@load "reusefiles/savedmodels/bounded_morecoherencematchingepoch20" model
newincomodel = model

@load "reusefiles/savedmodels/more_incoherentepoch20" model
oldincomodel = model

models = [newincomodel, oldincomodel]
modellabels = ["newincomodel", "oldincomodel"]
meas = logrange(40, 250, 3)

resultsdata = compare_models_from_thresholds(models, modellabels, meas, 2, 16, 28^2)

#recoverythreshold_fromrandomimage(model, model.decoder, logdata, 16, 28^2, savefile="reusefiles/experiment_data/scatter_recoveries_v2.BSON")

#train_incoherentVAE_onMNIST(vaelossfn=VAEloss_boundedcoherence, numepochs=20, label="bounded_morecoherencematching", α=1.0f3, λ=1.0f-2, logginglossterms=false)

#Other experiment: find threshold for many images

include("reusefiles/relaxedrecovery.jl")

layerdims = getlayerdims(model.decoder)

plot_MNISTrecoveries_bynumber_bymeasurementnumber_fast(model, model.decoder, [512], [5], layerdims)