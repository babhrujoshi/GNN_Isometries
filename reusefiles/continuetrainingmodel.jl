using Flux
using Flux: params, DataLoader
using MLDatasets
using BSON: @load

include("./vaemodel.jl")

@load "./reusefiles/models/tinyv3intrain6" model opt

include("./trainloops.jl")


loss = vaeloss(model, 1.0f0, 0.001f0)
traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
testdata = reshape(MNIST(Float32, :test).features[:, :, 1:end], 28^2, :)
trainloader = DataLoader(traindata, batchsize=32)
validateloader = DataLoader(testdata, batchsize=32)


trainvalidatelognsave(loss, model, params(model), trainloader, validateloader, opt, 20, "./reusefiles/models/", "./reusefiles/logs/", label="tinyv3phase0", loginterval=3, saveinterval=100)
