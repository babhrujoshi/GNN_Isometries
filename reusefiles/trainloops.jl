using Flux: train!, pullback
using ProgressMeter: Progress, next!
using TensorBoardLogger
using BSON: @save


"""
loss: loss function taking a data batch as argument.
data: an itterable that feeds data vectors matching model input when itterated upon
"""

function trainandgetloss!(lossfn, pars, dataloader)

    for x_batch in dataloader # pullback function returns the result (loss) and a pullback operator (back)
        loss, back = pullback(trainable_params) do
            lossfn(x_batch)
        end
        # Feed the pullback 1 to obtain the gradients and update then model parameters
        gradients = back(1.0f0)
        Flux.Optimise.update!(opt, pars, gradients)
        if isnan(loss)
            break
        end
        acc_loss += loss
    end

end

function trainlognsave(lossfn, model, pars::Params, data, opt::Flux.Optimise.AbstractOptimiser, numepochs, save_dir, tblogdir)
    # The training loop for the model
    with_logger(TBLogger(tblogdir)) do
        numbatches = length(data)
        
        for epoch_num = 1:numepochs
            acc_loss = 0.0f0
            loss = 0.0f0
            
            trainandgetloss!(lossfn,pars, data)

            avg_loss = acc_loss / numbatches
            @info epoch_num avg_loss
            @save string(save_dir, "_epoch_", epoch_num) model
        end
        println("Training complete!")
    end
end

