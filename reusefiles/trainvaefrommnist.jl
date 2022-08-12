using Flux
using Flux: params, DataLoader
using MLDatasets

include("./vaemodel.jl")
include("./trainloops.jl")

function trainvaefromMNIST(; label="")
    vaemodel = makevae()
    loss = vaeloss(vaemodel, 1.0f0, 0.01f0)
    traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end] |> gpu, 28^2, :)
    testdata = reshape(MNIST(Float32, :test).features[:, :, 1:end] |> gpu, 28^2, :)
    trainloader = DataLoader(traindata, batchsize=32, shuffle=true)
    testloader = DataLoader(testdata, batchsize=32, shuffle=true)
    trainvalidatelognsave(loss, vaemodel, params(vaemodel), trainloader, testloader, Flux.Optimise.ADAM(0.0001), 25, "./reusefiles/models/", "./reusefiles/logs/", saveinterval=50, validateinterval=5, label=label)
end

#for i in 1:20
#    trainvaefromMNIST(label=string("run", i))
#end

@load "./reusefiles/models/run1_final_epoch_25" model opt

loss = vaeloss(model, 1.0f0, 0.01f0)
traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end] |> gpu, 28^2, :)
testdata = reshape(MNIST(Float32, :test).features[:, :, 1:end] |> gpu, 28^2, :)
trainloader = DataLoader(traindata, batchsize=32, shuffle=true)
testloader = DataLoader(testdata, batchsize=32, shuffle=true)
trainvalidatelognsave(loss, model, params(model), trainloader, testloader, opt, 200, "./reusefiles/models/", "./reusefiles/logs/", saveinterval=10, validateinterval=5, label="randinitcontinued")
