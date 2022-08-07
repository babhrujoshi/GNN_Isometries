using Flux
using Flux: train!, pullback
using ProgressLogging
using Logging
using TensorBoardLogger
using BSON: @save
using CUDA
using RollingFunctions

include("./runningavg.jl")

"""
loss: loss function taking a data batch as argument.
data: an itterable that feeds data vectors matching model input when itterated upon
"""

function trainandgetloss!(lossfn, pars, dataloader, opt::Flux.Optimise.AbstractOptimiser)
    nextelt, getloss = runningavg()
    numelts = length(dataloader)*32
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
        nextelt(loss/numelts)
    end
    getloss()
end

function evaluateloss(lossfn, data)
    nextelt, getloss = runningavg()
    for x_batch in data
        nextelt(lossfn(x_batch))
    end
    getloss()
end


function trainlognsave(lossfn, model, pars::Flux.Params, data, opt::Flux.Optimise.AbstractOptimiser, numepochs, save_dir, tblogdir; save_interval=10)
    # The training loop for the model
    with_logger(TBLogger(tblogdir)) do
        #numbatches = length(data)
        
        for epoch_num in 1:numepochs
            
            loss = trainandgetloss!(lossfn,pars, data, opt)

            #avg_loss = loss / numbatches

            @info "epoch log" loss

            if epoch_num%save_interval == 0
                @save string(save_dir, "_epoch_", epoch_num) model
            end

        end
        @save string(save_dir, "_final_epoch_", numepochs) model
    end
    @info "training complete!"
end


function trainvalidatelognsave(lossfn, model, pars::Flux.Params, traindata, validatedata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; saveinterval=10, validateinterval = 10)
    # The training loop for the model
    tblogger = TBLogger(tblogdir)
        #numbatches = length(data)
        
        @progress for epochnum in 1:numepochs
            
            loss = trainandgetloss!(lossfn,pars, traindata, opt)

            with_logger(tblogger) do
                @info "epoch log" loss
                if epochnum % validateinterval == 0
                    @info "validating" epoch = epochnum evaluateloss(lossfn, validatedata)
                end
            end

            if epochnum%saveinterval == 0
                @save string(savedir, "_epoch_", epochnum) model
            end

        end
        with_logger(tblogger) do
            @info "validating" epoch = numepochs evaluateloss(lossfn, validatedata)
            @save string(savedir, "_final_epoch_", numepochs) model
        end
    @info "training complete!"
end

