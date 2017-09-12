#Module for either generating data or exporting from an existing dataset

module DataExamples

export get_Waveform, get_BreastCancer

function get_BreastCancer()
    data = readdlm("../data/Processed_BreastCancer.data",',')
    X = convert(Array{Float64,2},data[:,1:end-1])
    y = convert(Array{Float64,1},data[:,end])
    nSamples = size(X,1)
    shuffling = true
    DatasetName = "BreastCancer"
    Z = hcat(X,y)
    if shuffling
      Z = Z[shuffle(collect(1:nSamples)),:] #Shuffle the data
    end
    (Z[:,1:end-1],Z[:,end],DatasetName)
end

function get_Waveform()
    data = readdlm("../data/Processed_Waveform.data",',')
    X = convert(Array{Float64,2},data[:,1:end-1])
    y = convert(Array{Float64,1},data[:,end])
    nSamples = size(X,1)
    shuffling = false
    DatasetName = "Waveform"
    Z = hcat(X,y)
    if shuffling
      Z = Z[shuffle(collect(1:nSamples)),:] #Shuffle the data
    end
    (Z[:,1:end-1],Z[:,end],DatasetName)
end

end #end of module
