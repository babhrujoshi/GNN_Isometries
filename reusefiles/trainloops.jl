using Flux
using Flux: train!, pullback
using ProgressLogging
using Logging
using TensorBoardLogger
using BSON: @save
using CUDA
using RollingFunctions

#include("./runningavg.jl")

"""
loss: loss function taking a data batch as argument.
data: an itterable that feeds data vectors matching model input when itterated upon
"""

function trainandgetloss!(lossfn, pars, dataloader, opt::Flux.Optimise.AbstractOptimiser)
    #nextelt, getloss = runningavg()
    accloss = 0.0f0
    numelts = length(dataloader) * 32
    for x_batch in dataloader # pullback function returns the result (loss) and a pullback operator (back)
        loss, back = pullback(pars) do
            lossfn(x_batch)
        end
        # Feed the pullback 1 to obtain the gradients and update then model parameters
        gradients = back(1.0f0)
        Flux.Optimise.update!(opt, pars, gradients)
        if isnan(loss)
            continue
        end
        accloss += loss
    end
    accloss / numelts
end

function evaluateloss(lossfn, data)
    #nextelt, getloss = runningavg()
    accloss = 0.0f0
    numelts = length(data) * 32
    for x_batch in data
        accloss += lossfn(x_batch)
    end
    accloss / numelts
end


function trainlognsave(lossfn, model, pars::Flux.Params, data, opt::Flux.Optimise.AbstractOptimiser, numepochs, save_dir, tblogdir; save_interval=10)
    # The training loop for the model
    with_logger(TBLogger(tblogdir)) do
        #numbatches = length(data)

        for epoch_num in 1:numepochs

            loss = trainandgetloss!(lossfn, pars, data, opt)

            #avg_loss = loss / numbatches

            @info "epoch log" loss

            if epoch_num % save_interval == 0
                @save string(save_dir, "_epoch_", epoch_num) model
            end

        end
        @save string(save_dir, "_final_epoch_", numepochs) model
    end
    @info "training complete!"
end


function trainvalidatelognsave(lossfn, model, pars::Flux.Params, traindata, validatedata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; saveinterval=40, loginterval=10, label="")
    # The training loop for the model
    tblogger = TBLogger(tblogdir)
    saveindex = 0

    function savemodel()
        @save string(savedir, label, "intrain", saveindex) model opt
        saveindex += 1
    end

    #numbatches = length(data)
    for epochnum in 1:numepochs, (step, x_batch) in enumerate(traindata)

        loss, back = pullback(pars) do
            lossfn(x_batch)
        end
        gradients = back(1.0f0)
        Flux.Optimise.update!(opt, pars, gradients)

        if step % loginterval == 0
            with_logger(tblogger) do
                @info "loss" loss
            end
        end

        if step % saveinterval == 0
            savemodel()
            with_logger(tblogger) do
                @info "validation" evaluateloss(lossfn, validatedata)
            end
            @save string(savedir, label, "_epoch_", epochnum) model opt
        end


    end
    savemodel()
    @info "training complete!"
end
