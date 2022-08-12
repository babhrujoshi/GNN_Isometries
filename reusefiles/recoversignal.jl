using Flux
using MLDatasets
using Images
using BSON: @load
using LinearAlgebra
using Plots

include("./CompressedSensing.jl")
include("./vaemodel.jl")

imgToRecover = MNIST(Float32, :test).features[:,:,2]
colorview(Gray, imgToRecover)'

@load "./reusefiles/savedmodels/vae100epoch" model

imgToRecover = reshape(imgToRecover, :)

F = samplefourierwithoutreplacement(28^2, 28^2)

meas = F* imgToRecover

function show(sig)
    plot(colorview(Gray, reshape(sig, 28,28)'))
end

code = model.encoder(imgToRecover)[1]

code = 3.0*randn(20)

show(model.decoder(code))

recovery = recoversignal(meas,F,model.decoder, 20, initcode = copy(code), tblogdir="./reusefiles/logs/", out_toggle = 1000, opt = Flux.Optimise.ADAM(20.0), max_iter = 10_000)

show(recovery)