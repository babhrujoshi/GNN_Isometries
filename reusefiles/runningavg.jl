
function runningavg()
    currentavg = 0
    numprevsteps= 0.0
    function nextavg(nextelt)
        currentavg = currentavg*(numprevsteps/(numprevsteps+1.0)) + nextelt*(1.0/(numprevsteps+1.0))
        numprevsteps = numprevsteps+1.0
        nothing
    end
    function getloss()
        currentavg
    end
    (nextavg, getloss)
end