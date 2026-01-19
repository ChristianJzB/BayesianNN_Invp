
from ml_collections import ConfigDict
import numpy as np

def nv_experiment(N=50,hidden_layers=3,num_neurons=200,batch_size=160,kl=2,dgala=True,marginal=False):
    config = ConfigDict()
    config.model_type = "dgala"

    config.paths = ConfigDict()
    config.paths.base_dir = "./navier_stokes"
    config.paths.model_dir = f"{config.paths.base_dir}/models"
    config.paths.results_dir = f"{config.paths.base_dir}/results"

    # These will be dynamically built later based on hyperparams
    config.paths.nn_model_file = None
    config.paths.nn_mcmc_results_file = None
    config.paths.nn_da_mcmc_results_file = None

    config.paths.dgala_model_file = None
    config.paths.dgala_mcmc_results_file = None
    config.paths.dgala_da_mcmc_results_file = None


    # Weights & Biases
    config.wandb = wandb = ConfigDict()
    wandb.project = "Experiment_NV"
    wandb.name = "Vorticity"
    wandb.tag = None

    # General settings
    config.nn_model = "MDNN"  # Options: "NN", "WRF", "MDNN"
    config.lambdas = {"nvs":1, "cond":1, "w0":1, "phi":1}

    # Model-specific settings
    config.model = ConfigDict()
    config.model.input_dim = 3 + 2
    config.model.hidden_dim = num_neurons
    config.model.num_layers = hidden_layers
    config.model.out_dim = 2
    config.model.activation = "tanh"

    # Weight-Random-Factorization
    #config.reparam = ConfigDict({"type":"weight_fact","mean":1.0,"stddev":0.1})

     # Periodic embeddings
    config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":200,"exclude_last_n":2})

    # Navier Stokes Config
    config.nu = 1e-2
    config.time_domain = 2

    # Training settings
    config.learning_rate = 0.001
    config.decay_rate = 0.9
    config.alpha = 0.9  # For updating loss weights
    config.iterations = 5000
    config.start_scheduler = 0.1
    config.weights_update = 250
    config.scheduler_step = 1000 #2000

    config.chunks = 16
    config.points_per_chunk = N
    config.batch_size = batch_size
    #config.batch_ic = 16*

    # For deep Galerkin- initial conditions
    config.d = 5
    config.tau = np.sqrt(2)
    config.KL_expansion =  kl
    config.dim_initial_condition = 128
    config.samples_size_initial = 1000

    # DeepGala
    config.dgala = dgala
    config.marginal=marginal

    return config

def nv_inverse_problem(model_type,verbose,dgala=True,mcmc=False,da=True,marginal_approximation=False,alpha_eval=True,kl=2,
                            proposal="random_walk",mcmc_samples = 1_000_000,da_eval = 5_000, repeat =3):
    config = ConfigDict()
    config.model_type = model_type
    config.verbose = verbose

    config.dgala = dgala

    config.mcmc = mcmc
    config.da = da
    config.marginal_approximation = marginal_approximation
    config.alpha_eval = alpha_eval

    config.seed = 1234

    # Inverse problem parameters
    config.KL_expansion = kl
    config.noise_level = 1e-2
    config.num_observations = 6
    
    # Num Solver Config
    config.fs_n = 128
    config.fs_T = 2
    config.fs_steps = 5e-4

    config.proposal = proposal
    config.proposal_variance = 1e-1
    config.uniform_limit = 2
    config.prior_type = "uniform"
    config.samples = mcmc_samples
    
    config.repeat = repeat

    # Delayed Acceptance
    config.iter_mcmc = mcmc_samples
    config.iter_da = da_eval

    return config
