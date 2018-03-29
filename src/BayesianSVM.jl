if !isdefined(:KernelFunctions); include("KernelFunctions.jl"); end;
if !isdefined(:CustomKMeans); include("AFKMC2.jl"); end;

module BayesianSVM

using KernelFunctions
using CustomKMeans
using Distributions
using StatsBase
using PyPlot

export BSVM

#Corresponds to the BSVM model
type BSVM
  #Data
  X #Feature vectors
  y #Labels (-1,1)
  #Stochastic parameters
  Stochastic::Bool
    nSamplesUsed::Int64 #Size of the minibatch used
    κ_s::Float64 #Parameters for decay of learning rate (iter + κ)^-τ in case adaptative learning rate is not used
    τ_s::Float64
  #Non linear parameters
  NonLinear::Bool
  Sparse::Bool
    kernels::Array{Kernel,1} #Kernels function used
    γ::Float64 # Regularization parameter of the noise
    m::Int64 #Number of inducing points
  #Autotuning parameters
  Autotuning::Bool
    κ_Θ::Float64 #Parameters for decay of learning rate for the hyperparameter  (iter + κ)^-τ
    τ_Θ::Int64
    autotuningFrequency::Int64 #Frequency of update of the hyperparameter
  #Flag for adaptative learning rate for the SVI
  AdaptativeLearningRate::Bool
  #General Parameters for training
  Intercept::Bool
  ϵ::Float64 #Desired precision (on ||β(t)-β(t-1)||)
  nEpochs::Int64 #Maximum number of iterations
  β_init::Array{Float64,1} #Initial value for β
  smoothingWindow::Int64
  VerboseLevel::Int64
  Storing::Bool #Store values for debugging and visualization purposes
    StoringFrequency::Int64 #Every X steps
    StoredValues::Array{Float64,2}
    StoredDerivativesELBO::Array{Float64,2}
  #Functions
  Kernel_function::Function #kernel function associated with the model
  Train::Function #Model train for a certain number of iterations
  Predict::Function
  PredictProba::Function
  ELBO::Function
  DerivativesELBO::Function
  Plotting::Function
  Update::Function

  #Parameters learned with training
  nSamples::Int64 # Number of data points
  nFeatures::Int64 # Number of features
  μ::Array{Float64,1} # Mean for variational distribution
  η_1::Array{Float64,1} #Natural Parameter #1
  ζ::Array{Float64,2} # Covariance matrix of variational distribution
  η_2::Array{Float64,2} #Natural Parameter #2
  α::Array{Float64,1} # Distribution parameter of the GIG distribution of the latent variables
  invΣ::Array{Float64,2} #Inverse Prior Matrix for the linear case
  invK::Array{Float64,2} #Inverse Kernel Matrix for the nonlinear case
  invKmm::Array{Float64,2} #Inverse Kernel matrix of inducing points
  Ktilde::Array{Float64,1} #Diagonal of the covariance matrix between inducing points and generative points
  κ::Array{Float64,2} #Kmn*invKmm
  inducingPoints::Array{Float64,2} #Inducing points coordinates for the Big Data GP
  top #Storing matrices for repeated predictions (top and down are numerator and discriminator)
  down
  MatricesPrecomputed::Bool #Flag to know if matrices needed for predictions are already computed or not
  ρ_s::Float64 #Learning rate for CAVI
  g::Array{Float64,1} # g & h are expected gradient value for computing the adaptive learning rate
  h::Float64
  ρ_Θ::Float64 # learning rate for auto tuning
  initialized::Bool
  evol_β::Array{Float64,2} #Store the betas for smooth convergence criterium


  #Constructor
  function BSVM(X,y;Stochastic::Bool=false,Sparse::Bool=true,NonLinear::Bool=true,AdaptativeLearningRate::Bool=true,Autotuning::Bool=false,
                                  nEpochs::Int64 = 2000,batchSize::Int64=-1,κ_s::Float64=1.0,τ_s::Int64=100,
                                  kernels=0,γ::Float64=1e-3,m::Int64=0,κ_Θ::Float64=1.0,τ_Θ::Int64=100,autotuningfrequency::Int64=10,
                                  Intercept::Bool=false,ϵ::Float64=1e-5,β_init=[0.0],smoothingWindow::Int64=10,
                                  Storing::Bool=false,StoringFrequency::Int64=1,VerboseLevel::Int64=0)
    iter = 1
    if kernels == 0 && NonLinear
      warn("No kernel indicated, a rbf kernel function with lengthscale 1 is used")
      kernels = [Kernel("rbf",1.0,params=1.0)]
    end
    #Verification of consistency of the model
    this = new(X,y,Stochastic,batchSize,κ_s,τ_s,NonLinear,Sparse,kernels,γ,m,Autotuning,κ_Θ,τ_Θ,autotuningfrequency,AdaptativeLearningRate,Intercept,ϵ,nEpochs,β_init,smoothingWindow,VerboseLevel,Storing,StoringFrequency)
    if !ModelVerification(this,size(this.X),size(this.y))
      return
    end
    this.initialized = false
    if NonLinear
      this.top = 0
      this.down = 0
      MatricesPrecomputed = false
      this.Kernel_function = function(X1,X2)
          dist = 0
          for i in 1:size(this.kernels,1)
            dist += this.kernels[i].coeff*this.kernels[i].compute(X1,X2)
          end
          return dist
      end
      this.Train = function()
          TrainBSVM(this);
        end

      if Sparse
        this.Predict = function(X_test)
            SparsePredict(X_test,this)
          end
        this.PredictProba = function(X_test)
            SparsePredictProb(X_test,this)
          end
         #if this.Stochastic
        #     this.ELBO = function(X,y)
        #         NoisySparseELBO(this,y)
        #       end
        #  else
            this.ELBO = function()
                SparseELBO(this)
              end
        #   end
          this.DerivativesELBO = function() #Not implemented yet
              return 0
          end
      else
        this.Predict = function(X_test)
            NonLinearPredict(X_test,this)
          end
        this.PredictProba = function(X_test)
            NonLinearPredictProb(X_test,this)
          end
        this.ELBO = function()
            NonLinearELBO(this)
          end
        this.DerivativesELBO = function()
            DerivativesNonLinearELBO(this.y,this.μ,this.ζ,this.α,this.invK,this.Autotuning ? this.J : eye(size(this.X,1)))
          end
      end

    else
      this.Predict = function(X_test)
          LinearPredict(X_test,this)
        end
      this.PredictProba = function(X_test)
          LinearPredictProb(X_test,this)
        end
      this.ELBO = function()
          LinearELBO(this)
        end
      this.DerivativesELBO = function()
          DerivativesLinearELBO(Diagonal(this.y)*this.X,this.μ,this.ζ,this.α,inv(this.invΣ))
        end
    end
    this.Plotting = function(;s::String="All")
        ParametersPlotting(this,option=s)
      end
    this.Update = function(iter)
        Update(this,this.X,this.y,iter)
      end
    return this
  end
  #end of constructor
end

#Function to check consistency of the different parameters and the possible correction of some of them in some cases
function ModelVerification(model::BSVM,XSize,ySize)
  if model.Intercept && model.NonLinear
    warn("Not possible to have intercept for the non linear case, removing automatically this option")
    model.Intercept = false
  end
  if XSize[1] != ySize[1]
    warn("There is a dimension problem with the data size(y) != size(X)")
    return false
  end
  if model.NonLinear && model.Sparse
      minpoints = 50
    if model.m > XSize[1]
        if XSize[1] < 3.0*minpoints
            warn("Wrong inducing point setting, using the full batch method")
            model.Sparse = false
            model.Stochastic = false
        else
            warn("There are more inducing points than actual points, setting it to 10% of the datasize (minimum of $minpoints points)")
            model.m = min(minpoints,XSize[1]÷10)
        end
    elseif model.m == 0
        if XSize[1] < 3.0*minpoints
            warn("Number of inducing points was not manually set, using the full batch method")
            model.Sparse = false
            model.Stochastic = false
        else
            warn("Number of inducing points was not manually set, setting it to 10% of the datasize (minimum of $minpoints points)")
            model.m = min(minpoints,XSize[1]÷10)
        end
    end
  end
  if !model.Stochastic && model.smoothingWindow > 1
      # warn("Smoothing Windows larger than one is unnecessary for a non-stochastic optimization, setting it to one.")
      model.smoothingWindow = 1;
  end
  if model.Sparse && !model.NonLinear
    warn("Model cannot be sparse and linear at the same time, assuming linear model")
    model.Sparse = false;
  end
  if model.γ <= 0
    warn("Gamma should be strictly positive, setting it to default value 1.0e-3")
    model.γ = 1e-3
  end
  if model.nSamplesUsed == -1 && model.Stochastic
    warn("No batch size has been given, stochastic option has been removed")
    model.Stochastic = false
  end
  return true
end

function TrainBSVM(model::BSVM)
  model.nSamples = length(model.y)
  model.nFeatures = model.NonLinear ? (model.Sparse ? model.m : length(model.y)) : size(model.X,2)

  if model.VerboseLevel > 0
    println("Starting training of data of size $((model.nSamples,size(model.X,2))), using the"*(model.Autotuning ? " autotuned" : "")*(model.Stochastic ? " stochastic" : "")*(model.NonLinear ? " kernel" : " linear")*" method"
    *(model.AdaptativeLearningRate ? " with adaptative learning rate" : "")*(model.Sparse ? " with inducing points" : ""))
  end

  #Initialization of the variables
  if model.Intercept
    model.nFeatures += 1
    model.X = [ones(Float64,model.nSamples) model.X]
  end
  if !model.initialized
    if model.β_init[1] == 0 || length(model.β_init) != nFeatures
      if model.VerboseLevel > 2
        warn("Initial mean of the variational distribution is sampled from a multinormal distribution")
      end
      model.μ = randn(model.nFeatures)
    else
      model.μ = model.β_init
    end
    #Necessary to initialize only for first computation of the ELBO
    model.α = abs.(rand(model.nSamples))
    model.ζ = eye(model.nFeatures)
    #Creation of the Kernel Matrix and its inverse in the different cases as well as the prior
    if model.NonLinear
      if !model.Sparse
        model.invK = inv(Symmetric(CreateKernelMatrix(model.X,model.Kernel_function) + model.γ*eye(model.nFeatures),:U))
      end
      if model.Sparse
        model.inducingPoints = KMeansInducingPoints(model.X,model.m,10)
        model.invKmm = Matrix(Symmetric(inv(CreateKernelMatrix(model.inducingPoints,model.Kernel_function)+model.γ*eye(model.nFeatures))))
        Knm = CreateKernelMatrix(model.X,model.Kernel_function,X2=model.inducingPoints)
        model.κ = Knm*model.invKmm
        model.Ktilde = CreateDiagonalKernelMatrix(model.X,model.Kernel_function) + model.γ*ones(model.nSamples) - squeeze(sum(model.κ.*Knm,2),2) #diag(model.κ*transpose(Knm))
      end
    elseif !model.NonLinear
      model.invΣ =  (1.0/model.γ)*eye(model.nFeatures)
    end
    if (model.nSamplesUsed <= 0 || model.nSamplesUsed > model.nSamples)
      model.nSamplesUsed = model.nSamples
    end
    #Initialization of the natural parameters
    model.η_2 = -0.5*inv(model.ζ)
    model.η_1 = 2*model.η_2*model.μ
    if model.AdaptativeLearningRate && model.Stochastic
      batchindices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false)
      Z = model.NonLinear ? Diagonal(model.y) : Diagonal(model.y)*model.X
      (grad_1,grad_2) = NaturalGradientELBO(model.α,(model.NonLinear && model.Sparse) ? Z*model.κ : Z, model.NonLinear ? (model.Sparse ? model.invKmm : model.invK) : model.invΣ,model.nSamples/model.nSamplesUsed)
      model.τ_s = model.nSamplesUsed
      model.g = vcat(grad_1,reshape(grad_2,size(grad_2,1)^2))
      model.h = norm(vcat(grad_1,reshape(grad_2,size(grad_2,1)^2)))^2
    end
    model.ρ_Θ = model.Autotuning? (1+model.τ_Θ)^(-model.κ_Θ) : 1.0;
    model.ρ_s = model.Stochastic ? (model.AdaptativeLearningRate ? dot(model.g,model.g)/model.h : (1+model.τ_s)^(-model.κ_s)) : 1.0
    if model.Storing
      # Storing trace(ζ),ELBO,max(|α|),ρ_s,ρ_Θ/ρ_γ,||Θ||/γ
      model.StoredValues = zeros(model.nEpochs,6)
      model.StoredDerivativesELBO = zeros(model.nEpochs,4)
      model.StoredValues[1,:] = [model.ELBO(),trace(model.ζ),mean(model.μ),mean(model.α),model.γ,model.Autotuning ? model.ρ_Θ : 0.0]
      model.StoredDerivativesELBO[1,:] = model.DerivativesELBO()
    end
    model.initialized = true
    model.down = 0
    model.top = 0
    model.MatricesPrecomputed = false
  end
  evol_β = zeros(model.nEpochs,model.nFeatures)
  evol_β[1,:] = model.μ
  evol_ELBO = zeros(model.nEpochs)
  evol_ELBO[1] = model.ELBO()

  batchindices = collect(1:model.nSamples)
  current = evol_ELBO[1]
  conv = Inf #Initialization of the Convergence value
  iter::Int64 = 1
  ##End of Initialization of the parameters
  if model.VerboseLevel > 1
    println("Iteration $iter / $(model.nEpochs) (max)")
    println("Convergence : $conv, ELBO : $current")
  end
  #Two criterions for stopping, number of iterations or convergence
  while iter < model.nEpochs && conv > model.ϵ
    #Print some of the parameters
    model.Update(iter)
    iter += 1
    evol_β[iter,:] = model.μ
    evol_ELBO[iter] = model.ELBO()
    smooth_1 = mean(evol_ELBO[max(1,iter-2*model.smoothingWindow):iter-1,:]);smooth_2 = mean(evol_ELBO[max(2,iter-2*model.smoothingWindow+1):iter,:]);
    conv = abs.(smooth_1-smooth_2)/abs.(smooth_1)
    #smooth_1 = mean(evol_β[max(1,iter-2*model.smoothingWindow):iter-1,:],1);smooth_2 = mean(evol_β[max(2,iter-2*model.smoothingWindow+1):iter,:],1);
    #conv = norm(smooth_1/norm(smooth_1)-smooth_2/norm(smooth_2))

    #prev = current
    # if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
        current_ELBO = evol_ELBO[iter]
    # end
    #conv = abs(current-prev)
    if model.Storing && iter%model.StoringFrequency == 0
      if model.NonLinear && model.Stochastic && ((!model.Autotuning && iter<=2) || (model.Autotuning && ((iter-1)%model.autotuningFrequency == 0)))
        println("Recomputing Kernel matrices")
        model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(model.X,model.Kernel_function)+model.γ*eye(model.nSamples)),:U))
        if model.Autotuning
          model.J = CreateKernelMatrix(model.X,deriv_rbf,model.Θ)
        end
      end
      model.StoredValues[iter÷model.StoringFrequency,:] = [model.ELBO(),trace(model.ζ),mean(model.μ),mean(model.α),model.γ,model.Autotuning ? model.ρ_Θ : 0.0,]
      model.StoredDerivativesELBO[iter÷model.StoringFrequency,:] = model.DerivativesELBO()
      #println(model.StoredDerivativesELBO[iter÷model.StoringFrequency,:])
    end
    if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
      println("Iteration $iter / $(model.nEpochs) (max)")
      println("Convergence : $conv, ELBO : $current_ELBO")
      if model.Autotuning
        println("Gamma : $(model.γ)")
        for i in 1:size(model.kernels,1)
          println("(Coeff,Parameter) for kernel $i : $((model.kernels[i].coeff,(model.kernels[i].Nparams > 0)? model.kernels[i].param : 0))")
        end
        println("rho theta : $(model.ρ_Θ)")
      end
    end
  end
  if model.VerboseLevel > 0
    println("Training ended after $iter iterations")
  end
  if model.Storing
    model.StoredValues = model.StoredValues[1:iter÷model.StoringFrequency,:];
    model.StoredDerivativesELBO = model.StoredDerivativesELBO[1:iter÷model.StoringFrequency,:];
    model.evol_β = evol_β[1:iter,:]
  end
  return model
end



function Update(model::BSVM,X::Array{Float64,2},y::Array{Float64,1},iter::Int64) #Coordinates ascent of the parameters
    if model.Stochastic
      batchindices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false)
    else
      batchindices = collect(1:model.nSamples)
    end
    model.top = 0; model.down = 0; model.MatricesPrecomputed = false;#Need to recompute the matrices
    #Definition of the Z matrix, different for everycase
    Z = model.NonLinear ? (model.Sparse ? Diagonal(y[batchindices])*model.κ[batchindices,:] : Diagonal(y[batchindices]) ) : (Diagonal(y[batchindices])*X[batchindices,:])
    #Computation of latent variables
    model.α[batchindices] = (1 - Z*model.μ).^2
    model.α[batchindices] +=  squeeze(sum((Z*model.ζ).*Z,2),2)
    if model.Sparse && model.NonLinear
      model.α[batchindices] += model.Ktilde[batchindices] #Cf derivation of updates
    end

    #Compute the natural gradient
    (grad_η_1,grad_η_2) = NaturalGradientELBO(model.α[batchindices],Z, model.NonLinear ? (model.Sparse ? model.invKmm : model.invK) : model.invΣ, model.Stochastic ? model.nSamples/model.nSamplesUsed : 1.0)

    #Compute the learning rate
    if model.AdaptativeLearningRate && model.Stochastic
      #Using the paper on the adaptive learning rate for the SVI (update from the natural gradients)
      model.g = (1-1/model.τ_s)*model.g + vcat(grad_η_1-model.η_1,reshape(grad_η_2-model.η_2,size(grad_η_2,1)^2))./model.τ_s
      model.h = (1-1/model.τ_s)*model.h +norm(vcat(grad_η_1-model.η_1,reshape(grad_η_2-model.η_2,size(grad_η_2,1)^2)))^2/model.τ_s
      model.ρ_s = norm(model.g)^2/model.h
      #if iter%1==0
      #  println("g : $(norm(model.g)^2), h : $(model.h), rho : $(model.ρ_s), tau : $(model.τ_s)")
      #end
      model.τ_s = (1.0 - model.ρ_s)*model.τ_s + 1.0
    elseif model.Stochastic
      #Simple model of learning rate
      model.ρ_s = (iter+model.τ_s)^(-model.κ_s)
    else
      #Non-Stochastic case
      model.ρ_s = 1.0
    end
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)

    #Autotuning part, only happens every $autotuningFrequency iterations
    if model.Autotuning && (iter%model.autotuningFrequency == 0)
      if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
        println("Before hyperparameter optimization ELBO = $(model.ELBO())")
      end
      model.ρ_Θ = (iter+model.τ_Θ)^(-model.κ_Θ)
      if model.NonLinear
        if model.Sparse
          UpdateHyperparameterSparse!(model)
          model.invKmm = Matrix(Symmetric(inv(CreateKernelMatrix(model.inducingPoints,model.Kernel_function)+model.γ*eye(model.m)),:U))
          model.κ = CreateKernelMatrix(model.X,model.Kernel_function,X2=model.inducingPoints)*model.invKmm
        else
          UpdateHyperparameterNonLinear!(model)
          model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(model.X,model.Kernel_function)+model.γ*eye(model.nSamples)),:U))
        end
      else
        model.γ = UpdateHyperparameterLinear(model)
        model.invΣ = (1/model.γ)*eye(model.nFeatures)
      end
      if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
        println("After hyperparameter optimization ELBO = $(model.ELBO())")
      end
    end
end;

function NaturalGradientELBO(α,Z,invPrior,stoch_coef)
  grad_1 =  stoch_coef*transpose(Z)*(1./sqrt.(α)+1)
  grad_2 = -0.5*(stoch_coef*transpose(Z)*Diagonal(1./sqrt.(α))*Z + invPrior)
  (grad_1,grad_2)
end

function UpdateHyperparameterLinear!(model) #Gradient ascent for γ, noise
    model.γ = model.γ + model.ρ_Θ*0.5*((trace(model.ζ)+norm(model.μ))/(model.γ^2.0)-model.nFeatures/model.γ)
end

function UpdateHyperparameterNonLinear!(model)#Gradient ascent for Θ , kernel parameters
    if model.invK == 0
      model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(model.X,model.Kernel_function)+model.γ*eye(model.nSamples)),:U))
    end
    NKernels = size(model.kernels,1)
    A = model.invK*model.ζ-eye(model.nFeatures)
    grad_γ = 0.5*(sum(model.invK.*A)+dot(model.μ,model.invK*model.invK*model.μ))
    if model.VerboseLevel > 2
      println("Grad gamma : $grad_γ")
    end
    model.γ = ((model.γ + model.ρ_Θ*grad_γ) < 0 ) ? model.γ/2 : (model.γ + model.ρ_Θ*grad_γ)
    #Update of both the coefficients and hyperparameters of the kernels
    if NKernels > 1 #If multiple kernels only update the kernel weight
      for i in 1:NKernels
        V = model.invK.*CreateKernelMatrix(model.X,model.kernels[i].compute)
        grad =  0.5*(sum(V.*A)+dot(model.μ,V*model.invK*model.μ))#update of the coeff
        if model.VerboseLevel > 2
          println("Grad kernel $i: $grad")
        end
        model.kernels[i].coeff =  ((model.kernels[i].coeff + model.ρ_Θ*grad) < 0 ) ? model.kernels[i].coeff/2 : (model.kernels[i].coeff + model.ρ_Θ*grad)
      end
    elseif model.kernels[1].Nparams > 0 #If only one update the kernel lengthscale
        V = model.invK*model.kernels[1].coeff*CreateKernelMatrix(model.X,model.kernels[1].compute_deriv)
        grad =  0.5*(sum(V.*A)+dot(model.μ,V*model.invK*model.μ))#update of the hyperparameter
        model.kernels[1].param =  ((model.kernels[1].param + model.ρ_Θ*grad) < 0 ) ? model.kernels[1].param/2 : (model.kernels[1].param + model.ρ_Θ*grad)
        if model.VerboseLevel > 2
          println("Grad kernel: $grad")
        end
    end
end

function UpdateHyperparameterSparse!(model)#Gradient ascent for Θ , kernel parameters #Not finished !!!!!!!!!!!!!!!!!!!!!!!!!!
  NKernels = size(model.kernels,1)
  A = eye(model.nFeatures)-model.invKmm*model.ζ
  B = model.μ*transpose(model.μ) + model.ζ
  Kmn = CreateKernelMatrix(model.inducingPoints,model.Kernel_function;X2=model.X)
  #Computation of noise constant
  Jnm = 0
  ι = (Jnm-model.κ)*model.invKmm
  grad_γ = -0.5*(sum(model.invKmm.*A) - dot(model.μ, transpose(model.μ)*model.invKmm*model.invKmm + 2*transpose(ones(model.nSamples)+1./sqrt.(model.α))*diagm(model.y)*ι)+
  dot(1./sqrt.(model.α),diag(model.κ*(B*transpose(ι)-transpose(Jnm)) + ι*(B*transpose(model.κ)-Kmn) )+ ones(model.nSamples)))
  if model.VerboseLevel > 2
    println("Grad gamma : $grad_γ")
  end
  model.γ = ((model.γ + model.ρ_Θ*grad_γ) < 0 ) ? (model.γ < 1e-7 ? model.γ : model.γ/2) : (model.γ + model.ρ_Θ*grad_γ)
  if NKernels > 1
    for i in 1:NKernels
      Jnm = CreateKernelMatrix(model.X,model.kernels[i].compute,X2=model.inducingPoints)
      Jnn = CreateDiagonalKernelMatrix(model.X,model.kernels[i].compute)
      Jmm = CreateKernelMatrix(model.inducingPoints,model.kernels[i].compute)
      ι = (Jnm-model.κ*Jmm)*model.invKmm
      V = model.invKmm*Jmm
      grad = -0.5*(sum(V.*A) - dot(model.μ, transpose(model.μ)*V*model.invKmm + 2*transpose(ones(model.nSamples)+1./sqrt.(model.α))*diagm(model.y)*ι) +
      dot(1./sqrt.(model.α),diag(model.κ*(B*transpose(ι)-transpose(Jnm)) + ι*(B*transpose(model.κ)-Kmn))+ Jnn))
      model.kernels[i].coeff =  ((model.kernels[i].coeff + model.ρ_Θ*grad) < 0 ) ? model.kernels[i].coeff/2 : (model.kernels[i].coeff + model.ρ_Θ*grad)
      if model.VerboseLevel > 2
        println("Grad kernel $i: $grad")
      end
    end
  elseif model.kernels[1].Nparams > 0 #Update of the hyperparameters of the KernelMatrix
    Jnm = model.kernels[1].coeff*CreateKernelMatrix(model.X,model.kernels[1].compute_deriv,X2=model.inducingPoints)
    Jnn = model.kernels[1].coeff*CreateDiagonalKernelMatrix(model.X,model.kernels[1].compute_deriv)
    Jmm = model.kernels[1].coeff*CreateKernelMatrix(model.inducingPoints,model.kernels[1].compute_deriv)
    ι = (Jnm-model.κ*Jmm)*model.invKmm
    V = model.invKmm*Jmm
    grad = -0.5*(sum(V.*A) - dot(model.μ, transpose(model.μ)*V*model.invKmm + 2*transpose(ones(model.nSamples)+1./sqrt.(model.α))*diagm(model.y)*ι) +
    dot(1./sqrt.(model.α),diag(model.κ*(B*transpose(ι)-transpose(Jnm)) + ι*(B*transpose(model.κ)-Kmn))+Jnn))
    model.kernels[1].param =  ((model.kernels[1].param + model.ρ_Θ*grad) < 0 ) ? model.kernels[1].param/2 : (model.kernels[1].param + model.ρ_Θ*grad)
    if model.VerboseLevel > 2
      println("Grad kernel: $grad, new param is $(model.kernels[1].param)")
    end
  end
end

function LinearELBO(model) #Compute the loglikelihood of the training data, ####-----Could be improved in algebraic form---####
    Z = Diagonal(model.y)*model.X
    ELBO = 0.5*(logdet(model.ζ)+logdet(model.invΣ)-trace(model.invΣ*model.ζ)-dot(model.μ,model.ζ*model.μ));
    for i in 1:model.nSamples
        ELBO += 1.0/2.0*log.(model.α[i]) + log.(besselk.(0.5,model.α[i])) + dot(vec(Z[i,:]),model.μ) + 0.5/model.α[i]*(model.α[i]^2-(1-dot(vec(Z[i,:]),model.μ))^2 - dot(vec(Z[i,:]),model.ζ*vec(Z[i,:])))
    end
    return ELBO
end

function DerivativesLinearELBO(Z,μ,ζ,α,Σ)
    (n,p) = size(Z)
    dζ = 0.5*(inv(ζ)-inv(Σ)-transpose(Z)*Diagonal(1./sqrt.(α))*Z)
    dμ = -inv(ζ)*μ + transpose(Z)*(1./sqrt.(α)+1)
    dα = zeros(n)
    for i in 1:n
      dα[i] = ((1-dot(Z[i,:],μ))^2 + dot(Z[i,:],ζ*Z[i,:]))/(2*(α[i])) - 0.5
    end
    γ = Σ[1,1]
    dγ = 0.5*((trace(ζ)+norm(μ))/(γ^2.0)-p/γ)
    return [trace(dζ),mean(dμ),mean(dα),dγ]
end


function NonLinearELBO(model)
  ELBO = 0.5*(logdet(model.ζ)+logdet(model.invK)-trace(model.invK*model.ζ)-dot(model.μ,model.invK*model.μ))
  for i in 1:model.nSamples
    ELBO += 0.25*log.(model.α[i])+log.(besselk.(0.5,sqrt.(model.α[i])))+model.y[i]*model.μ[i]+(model.α[i]-(1-model.y[i]*model.μ[i])^2-model.ζ[i,i])/(2*sqrt.(model.α[i]))
  end
  return ELBO
end

function DerivativesNonLinearELBO(y,μ,ζ,α,invK,J)
  n = length(y)
  dζ = 0.5*(inv(ζ)-invK-Diagonal(1./sqrt.(α)))
  dμ = -inv(ζ)*μ + Diagonal(y)*(1./sqrt.(α)+1)
  dα = zeros(n)
  for i in 1:n
    dα[i] = ((1-y[i]*μ[i])^2+ζ[i,i])/(2*α[i])-0.5
  end
  dΘ = 0.5*(trace(invK*J*(invK*ζ-1))+dot(μ,invK*J*invK*μ))
  return [trace(dζ),mean(dμ),mean(dα),mean(dΘ)]
end

function SparseELBO(model)
  ELBO = 0.0
  ELBO += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
  ELBO += -0.5*(sum(model.invKmm.*model.ζ)+dot(model.μ,model.invKmm*model.μ)) #trace replaced by sum
  ELBO += dot(model.y,model.κ*model.μ)
  ELBO += sum(0.25*log.(model.α) + log.(besselk.(0.5,sqrt.(model.α))))
  ζtilde = model.κ*model.ζ*transpose(model.κ)
  for i in 1:model.nSamples
    ELBO += 0.5/sqrt.(model.α[i])*(model.α[i]-(1-model.y[i]*dot(model.κ[i,:],model.μ))^2-(ζtilde[i,i]+model.Ktilde[i]))
  end
  return ELBO
end


function NoisySparseELBO(model,indices)
    n = length(indices)
    ELBO = 0.0
    ELBO += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO += -0.5*(sum(model.invKmm.*model.ζ)+dot(model.μ,model.invKmm*model.μ)) #trace replaced by sum
    ELBO += dot(model.y,model.κ*model.μ)
    ELBO += sum(0.25*log.(model.α[indices]) + log.(besselk.(0.5,sqrt.(model.α[indices]))))
    ζtilde = model.κ[indices,:]*model.ζ*transpose(model.κ[indices,:])
    for i in 1:n
      ELBO += 0.5/sqrt.(model.α[indices[i]])*(model.α[indices[i]]-(1-model.y[indices[i]]*dot(model.κ[indices[i],:],model.μ))^2-(ζtilde[i,i]+model.Ktilde[i]))
    end
    return ELBO
end


function ParametersPlotting(model::BSVM;option::String="All")
  if !model.Storing
    warn("Data was not saved during training, please rerun training with option Storing=true")
    return
  elseif isempty(model.StoredValues )
    warn("Model was not trained yet, please run a dataset before");
    return
  end
  figure("Evolution of model properties over time");
  iterations =  floor(Int64,1:size(model.evol_β,1))-1
  nsiterations = size(model.StoredValues,1);
  sparseiterations = floor(Int64,linspace(0,(nsiterations-1)*model.StoringFrequency,nsiterations))
  if option == "All"
    nFeatures = model.Autotuning ? 6 : 4;
    subplot(nFeatures÷2,2,1)
    plot(sparseiterations,model.StoredValues[:,1])
    ylabel("ELBO")
    subplot(nFeatures÷2,2,2)
    plot(sparseiterations,model.StoredValues[:,2])
    ylabel(L"Trace($\zeta$)")
    subplot(nFeatures÷2,2,3)
    plot(sparseiterations,model.StoredValues[:,3])
    ylabel(L"Mean($\mu$)")
    subplot(nFeatures÷2,2,4)
    plot(sparseiterations,model.StoredValues[:,4])
    ylabel(L"Mean($\alpha_i$)")
    if model.Autotuning
      subplot(nFeatures÷2,2,5)
      semilogy(sparseiterations,model.StoredValues[:,5])
      ylabel(model.NonLinear ? L"||\theta||" : L"\gamma")
      subplot(nFeatures÷2,2,6)
      semilogy(sparseiterations,model.StoredValues[:,6])
      ylabel(L"\rho_\theta")
    end
  elseif option == "dELBO"
    (nIterations,nFeatures) = size(model.StoredDerivativesELBO)
    if model.Sparse
        warn("ELBO derivatives not available yet for the Sparse Algorithm")
    end
    DerivativesLabels = ["d\zeta" "d\mu" "d\alpha" "d\theta"]
    for i in 1:4
      if i <= 3  || (i==4 && model.Autotuning)
        subplot(2,2,i)
        plot(sparseiterations,model.StoredDerivativesELBO[:,i])
        ylabel(DerivativesLabels[i])
      end
    end
elseif option == "Mu"
    plot(iterations,sqrt.(sumabs2(model.evol_β)))
    ylabel(L"Normalized $\mu$")
    xlabel("Iterations")
  elseif option == "ELBO"
    plot(iterations,model.StoredValues[:,2])
    ylabel("ELBO")
    xlabel("Iterations")
  else
    warn("Option not available, chose among those : All, dELBO, Mu, Autotuning, ELBO")
  end
  return;
end


function LinearPredict(X_test,model)
  return model.Intercept ? [ones(Float64,size(X_test,1)) X_test]*model.μ : X_test*model.μ
end

function LinearPredictProb(X_test,model)
  if model.Intercept
    X_test = [ones(Float64,size(X_test,1)) X_test]
  end
  n = size(X_test,1)
  predic = zeros(n)
  for i in 1:n
    predic[i] = cdf(Normal(),(dot(X_test[i,:],model.μ))/(dot(X_test[i,:],model.ζ*X_test[i,:])+1))
  end
  return predic
end

function NonLinearPredict(X_test,model)
  n = size(X_test,1)
  if model.top == 0
    model.top = model.invK*model.μ
  end
  k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
  return k_star*model.top
end

function NonLinearPredictProb(X_test,model)
  n = size(X_test,1)
  if model.down == 0
    if model.invK == 0
      model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(model.X,model.kernelfunction,model.Θ)+model.γ*eye(model.nSamples)),:U))
    end
    model.top = model.invK*model.μ
    model.down = -(model.invK+model.invK*model.ζ*model.invK)
  end
  ksize = model.nSamples
  predic = zeros(n)
  k_star = zeros(ksize)
  k_starstar = 0
  for i in 1:n
    for j in 1:ksize
      k_star[j] = model.Kernel_function(model.X[j,:],X_test[i,:])
    end
    k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
    predic[i] = cdf(Normal(),(dot(k_star,model.top))/(k_starstar + dot(k_star,model.down*k_star) + 1))
  end
  predic
end

function SparsePredictProb(X_test,model)
  n = size(X_test,1)
  ksize = model.m
  if model.down == 0
    if model.top == 0
      model.top = model.invKmm*model.μ
    end
    model.down = model.invKmm*(eye(ksize)+model.ζ*model.invKmm)
    model.MatricesPrecomputed = true
  end
  predic = zeros(n)
  k_star = zeros(ksize)
  k_starstar = 0
  for i in 1:n
    for j in 1:ksize
      k_star[j] = model.Kernel_function(model.inducingPoints[j,:],X_test[i,:])
    end
    k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])

    predic[i] = cdf(Normal(),(dot(k_star,model.top))/(k_starstar - dot(k_star,model.down*k_star) + 1))
  end
  predic
end


function SparsePredict(X_test,model)
  n = size(X_test,1)
  if model.top == 0
    model.top = model.invKmm*model.μ
  end
  k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.inducingPoints)
  return k_star*model.top
end

end #End Module
