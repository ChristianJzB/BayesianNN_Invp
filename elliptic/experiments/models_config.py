
from ml_collections import ConfigDict


def elliptic_neural_training(N=2500,hidden_layers=2,num_neurons=20,batch_size=10,kl=2,dgala=True,marginal=False):
    config = ConfigDict()
    config.model_type = "dgala"

    config.paths = ConfigDict()
    config.paths.base_dir = "./elliptic"
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
    config.wandb = ConfigDict()
    config.wandb.project = "Elliptic-IP"
    config.wandb.name = "MDNN"
    config.wandb.tag = None

    # Model settings
    config.nn_model = "MDNN"
    config.lambdas = {"elliptic": 1, "ubcl": 1, "ubcr": 1}

    config.model = ConfigDict()
    config.model.input_dim = 3
    config.model.hidden_dim = num_neurons
    config.model.num_layers = hidden_layers
    config.model.out_dim = 1
    config.model.activation = "tanh"
    config.KL_expansion = kl

    # Weight-Random-Factorization
    #config.reparam = ConfigDict({"type":"weight_fact","mean":1.0,"stddev":0.1})

    # Periodic embeddings
    #config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":20,"exclude_last_n":2})

    # Training settings
    config.seed = 42
    config.learning_rate = 0.001
    config.decay_rate = 0.95
    config.epochs = 5_000
    config.start_scheduler = 0.5
    config.scheduler_step = 100
    config.nn_samples = N
    config.batch_size = batch_size
    config.weights_update = 250
    config.alpha = 0.9
    # DeepGala
    config.dgala = dgala
    config.marginal=marginal

    return config

def elliptic_pigp_experiment(kernel_spatial ="MaternKernel52",kernel_parameter="SquaredExponential",N=10,spatial_points=6,kl=2,marginal=False):
    config = ConfigDict()
    config.model_type = "pigp"
    config.marginal=marginal

    config.paths = ConfigDict()
    config.paths.base_dir = "./elliptic"
    config.paths.model_dir = f"{config.paths.base_dir}/models"
    config.paths.results_dir = f"{config.paths.base_dir}/results"

    # These will be dynamically built later based on hyperparams
    config.paths.pigp_model_file = None
    config.paths.pigp_mcmc_results_file = None
    config.paths.pigp_da_mcmc_results_file = None

    # PIGP training config
    config.kernel_spatial = kernel_spatial
    config.kernel_parameter = kernel_parameter
    config.KL_expansion = kl
    config.observed_solutions = N
    config.observed_spatial_points = spatial_points
    return config

def elliptic_gp_experiment(kernel = "MaternKernel52",N=10,spatial_points=6,kl=2,marginal=False):
    config = ConfigDict()
    config.model_type = "gp"
    config.marginal=marginal

    config.paths = ConfigDict()
    config.paths.base_dir = "./elliptic"
    config.paths.model_dir = f"{config.paths.base_dir}/models"
    config.paths.results_dir = f"{config.paths.base_dir}/results"

    # These will be dynamically built later based on hyperparams
    config.paths.gp_model_file = None
    config.paths.gp_mcmc_results_file = None
    config.paths.gp_da_mcmc_results_file = None

    # GP training config
    config.kernel = kernel
    config.KL_expansion = kl
    config.observed_solutions = N
    config.observed_spatial_points = spatial_points
    return config


def elliptic_fem_experiment(kl=2):
    config = ConfigDict()
    config.model_type = "fem"

    config.paths = ConfigDict()
    config.paths.base_dir = "./elliptic"
    config.paths.model_dir = f"{config.paths.base_dir}/models"
    config.paths.results_dir = f"{config.paths.base_dir}/results"

    # These will be dynamically built later based on hyperparams
    config.fem_solver = 50
    config.KL_expansion = kl
    return config


def elliptic_inverse_problem(model_type,verbose,dgala=True,mcmc=False,da=True,marginal_approximation=False,alpha_eval=True,kl=2,
                            proposal="random_walk",mcmc_samples = 2_500_000,da_eval = 50_000, repeat =3):
    config = ConfigDict()
    config.model_type = model_type
    config.verbose = verbose

    config.dgala = dgala

    config.mcmc = mcmc
    config.da = da
    config.marginal_approximation = marginal_approximation
    config.alpha_eval = alpha_eval

    # Inverse problem parameters
    config.KL_expansion = kl
    config.noise_level = 1e-4
    config.num_observations = 6
    config.fem_solver = 50

    config.proposal = proposal
    config.proposal_variance = 1e-1
    config.prior_type = "uniform"
    config.uniform_limit = 2
    config.samples = mcmc_samples
    config.FEM_h = 50
    config.repeat = repeat

    # Delayed Acceptance
    config.iter_mcmc = mcmc_samples
    config.iter_da = da_eval


    return config