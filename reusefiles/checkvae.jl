using BSON: @load
using Flux

include("vaemodel.jl")

@load "./reusefiles/models/smallintrain6" model

using Plots
using Images
using MLDatasets


img = MNIST(Float32, :test).features[:, :, 32]

recodedimg = reshape(model(reshape(img, :)), 28, 28)

plot(
    plot(colorview(Gray, img)), plot(colorview(Gray, recodedimg))
)