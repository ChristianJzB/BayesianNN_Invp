import os
import sys
import json  # or import pickle if you prefer pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "elliptic"))  # Explicitly add Elliptic folder


from Base.lla import dgala
from Base.gp_base import MultioutputGP
from Base.utilities import *
from elliptic_files.train_elliptic import train_elliptic,train_dg_elliptic
from elliptic_files.utilities import *
from elliptic_files.FEM_Solver import FEMSolver
from elliptic_files.elliptic_pigp import Elliptic1DPIGP
from elliptic_files.elliptic_mcmc import EllipticMCMCDA
from torch.profiler import profile, record_function, ProfilerActivity


def build_paths(config):
    # Ensure directories exist
    model_type = getattr(config, "model_type", "model")

    os.makedirs(config.paths.model_dir, exist_ok=True)
    os.makedirs(config.paths.results_dir, exist_ok=True)

    if model_type == "dgala":
        # Build shared model ID
        aprox_type = "marginal" if config.marginal else "mean"
        model_id = (
            f"_hl{config.model.num_layers}"
            f"_hd{config.model.hidden_dim}"
            f"_s{config.nn_samples}"
            f"_b{config.batch_size}"
            f"_kl{config.KL_expansion}")

        # NN paths
        paths = {
            "dgala_mean_model": os.path.join(config.paths.model_dir, f"dgala_mean{model_id}.pth"),
            "dgala_marginal_model": os.path.join(config.paths.model_dir, f"dgala_marginal{model_id}.pth"),
            "dgala_mcmc": os.path.join(config.paths.results_dir, f"dgala_mcmc_{aprox_type}{model_id}" + "_{i}.npy"),
            "dgala_da": os.path.join(config.paths.results_dir, f"dgala_da_{aprox_type}{model_id}" + "_{i}.npy"),
            "dgala_da_chain": os.path.join(config.paths.results_dir, f"dgala_da_chain_{aprox_type}{model_id}" + "_{i}.npy"),
            "dgala_mean_proposal": os.path.join(config.paths.results_dir, f"dgala_mean_proposal_{model_id}.npy"),
            }
    
    elif model_type == "pigp":
        aprox_type = "marginal" if config.marginal else "mean"

        model_id = (
            f"_kernel_spatial{config.kernel_spatial}"
            f"_kernel_parameter{config.kernel_parameter}"
            f"_spatial{config.observed_spatial_points}"
            f"_nsol{config.observed_solutions}"
            f"_kl{config.KL_expansion}")
        
        paths ={"pigp_model": os.path.join(config.paths.model_dir,f"pigp{model_id}.pth"),
                "pigp_mcmc":os.path.join(config.paths.results_dir,f"pigp_mcmc_{aprox_type}{model_id}" + "_{i}.npy"),
                "pigp_da":os.path.join(config.paths.results_dir,f"pigp_da_{aprox_type}{model_id}" + "_{i}.npy"),
                "pigp_da_chain": os.path.join(config.paths.results_dir, f"pigp_da_chain_{aprox_type}{model_id}" + "_{i}.npy"),
                "pigp_mean_proposal":os.path.join(config.paths.results_dir, f"pigp_mean_proposal_{model_id}.npy")}
    
    elif model_type == "gp":
        aprox_type = "marginal" if config.marginal else "mean"
        model_id = (
            f"_kernel{config.kernel}"
            f"_spatial{config.observed_spatial_points}"
            f"_nsol{config.observed_solutions}"
            f"_kl{config.KL_expansion}")
        
        paths ={{"gp_model": os.path.join(config.paths.model_dir,f"gp{model_id}.pth"),
                 "gp_mcmc":os.path.join(config.paths.results_dir,f"gp_mcmc_{aprox_type}{model_id}" + "_{i}.npy"),
                 "gp_da_mcmc": os.path.join(config.paths.results_dir,f"gp_da_mcmc_{aprox_type}{model_id}" + "_{i}.npy"),
                 "gp_mean_proposal":os.path.join(config.paths.results_dir, f"gp_mean_proposal_{model_id}.npy"),}}
    
    elif model_type == "fem":
        model_id = (f"_kl{config.KL_expansion}")
        paths = {"fem_mcmc": os.path.join(config.paths.results_dir,f"FEM_mcmc{model_id}" + "_{i}.npy"),
                 "fem_mcmc_eval":os.path.join(config.paths.results_dir,f"FEM_mcmc_eval{model_id}" + "_{i}.npy")}

    return paths
    
def train_neural_network(config_experiment,path,device):
    model_specific = f"_mean_hl{config_experiment.model.num_layers}_hd{config_experiment.model.hidden_dim}_s{config_experiment.nn_samples}_b{config_experiment.batch_size}_kl{config_experiment.KL_expansion}"

    print(f"Running training with {config_experiment.nn_samples} samples...")
    config_experiment.wandb.name = config_experiment.model_type + model_specific
    config_experiment.model.input_dim = 1 + config_experiment.KL_expansion
    config_experiment.model.fourier_emb["embed_dim"] = config_experiment.model.hidden_dim
    config_experiment.model.fourier_emb["exclude_last_n"] = config_experiment.KL_expansion
    pinn_nvs = train_elliptic(config_experiment, device=device)

    print(f"Completed training model {config_experiment.model_type}{model_specific}.")

    if config_experiment.dgala:
        nn_model = torch.load(path["dgala_mean_model"], map_location=device)
        nn_model.eval()
        data_fit = deepgala_data_fit(config_experiment.nn_samples,config_experiment.KL_expansion,device)
        llp = dgala(nn_model)
        llp.fit(data_fit)
        llp.optimize_marginal_likelihood(max_iter=1000)
        clear_hooks(llp)
        torch.save(llp, path["dgala_marginal_model"])


def fit_deepgala(config_experiment,path,device):
    print(f"Starting DeepGaLA fitting for NN_s{config_experiment.nn_samples}")

    model_path = path["dgala_mean_model"]
    if os.path.exists(model_path):
            nn_model = torch.load(model_path, map_location=device)
            nn_model.eval()
    else: 
        train_neural_network(config_experiment,path,device)
        nn_model = torch.load(model_path, map_location=device)
        nn_model.eval()

    data_fit = deepgala_data_fit(config_experiment.nn_samples,config_experiment.KL_expansion,device)
    llp = dgala(nn_model)
    llp.fit(data_fit)

    sigma_trace, prior_trace, marglik_trace = llp.optimize_marginal_likelihood( max_iter=1000)

    # np.save(f"./Elliptic/models/sigma_trace_{config_experiment.nn_samples}.npy", sigma_trace)
    # np.save(f"./Elliptic/models/prior_trace_{config_experiment.nn_samples}.npy", prior_trace)
    # np.save(f"./Elliptic/models/marglik_trace_{config_experiment.nn_samples}.npy", marglik_trace)

    clear_hooks(llp)
    torch.save(llp, path["dgala_marginal_model"])


def train_pigp(config_experiment,path,device):    
    print(f"Training GP_s{config_experiment.observed_solutions}")
    data_training = pigp_training_data_generation(config_experiment.observed_solutions,config_experiment.observed_spatial_points,config_experiment.KL_expansion,device)

    elliptic_gp = Elliptic1DPIGP(data_training,kernel_spatial=config_experiment.kernel_spatial,
                                 kernel_parameter =config_experiment.kernel_parameter,device=device)
    elliptic_gp.train_gp()
    elliptic_gp.optimize_gp()

    torch.save(elliptic_gp, path["pigp_model"])

def train_gp(config_experiment,path,device):    
    print(f"Training GP_s{config_experiment.observed_solutions}")
    data_training = pigp_training_data_generation(config_experiment.observed_solutions,config_experiment.observed_spatial_points,config_experiment.KL_expansion,device)

    mine_gp = MultioutputGP(data_training,kernel =config_experiment.kernel)
    mine_gp.train_gp()
    mine_gp.optimize_gp()

    torch.save(mine_gp, path["gp_model"])

def get_models(config, config_model,paths, device):
    if config.model_type == "dgala":
        model_name = "dgala"
        if not config.marginal_approximation:
            model_path = paths.get("dgala_mean_model")

            if not os.path.exists(model_path):
                train_neural_network(config_model,paths,device)
            
            model = torch.load(paths["dgala_mean_model"], map_location=device)
            model.eval()

        else:
            model_path = paths.get("dgala_marginal_model")
            if not os.path.exists(model_path):
                fit_deepgala(config_model,paths,device)

            model = torch.load(paths["dgala_marginal_model"], map_location=device)
            model.model.set_last_layer("output_layer")
            
            model._device = device

    elif config.model_type == "pigp":
        model_name = "pigp"
        model_path = paths.get("pigp_model")
        if not os.path.exists(model_path):
            train_pigp(config_model,paths,device)
        
        model = torch.load(paths["pigp_model"], map_location=device)

    elif config.model_type == "gp":
        model_name = "gp"
        model_path = paths.get("gp_model")
        if not os.path.exists(model_path):
            train_gp(config_model,paths,device)
        model = torch.load(paths["gp_model"], map_location=device)

    elif config.model_type == "fem":
        model_name = "fem"
        model = FEMSolver(np.zeros(config.KL_expansion), vert=config.FEM_h, M=config.KL_expansion)

    return model_name,model

def get_sampler(model,config_experiment,device):
     # Step 3: Generate noisy observations for Inverse Problem
    obs_points, sol_test = generate_noisy_obs(obs=config_experiment.num_observations,
                                              std=np.sqrt(config_experiment.noise_level),
                                              nparam=config_experiment.KL_expansion,
                                              vert=config_experiment.fem_solver)
    
    fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h,
                           M = config_experiment.KL_expansion)

    elliptic_mcmcda = EllipticMCMCDA(model,fem_solver, 
                        observation_locations= obs_points, observations_values = sol_test, 
                        nparameters=config_experiment.KL_expansion,
                        marginal = config_experiment.marginal_approximation,
                        observation_noise=np.sqrt(config_experiment.noise_level),
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        proposal_type=config_experiment.proposal,
                        prior_type=config_experiment.prior_type,
                        uniform_limit = config_experiment.uniform_limit,
                        step_size=config_experiment.proposal_variance, 
                        device=device) 
    return elliptic_mcmcda,obs_points, sol_test



def flops_time(model,obs_points, sol_test,config_experiment,device):
        samples = samples_param(10_000, config_experiment.KL_expansion)
        
        obs_points = torch.tensor(obs_points,device = device, dtype = torch.float64)

        mcmc = EllipticMCMCDA(model,None, obs_points, sol_test, 
                              nparameters=config_experiment.KL_expansion,marginal=config_experiment.marginal_approximation,
                              observation_noise=np.sqrt(config_experiment.noise_level),device=device)

        table_res = {"total_cpu_time":[],"total_cuda_time":[],"total_flops":[]}
        for sample in samples:
            theta = torch.tensor(sample,device = device, dtype = torch.float64).reshape(-1)

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                mcmc.log_likelihood_outer(theta)
                                
            table_res["total_flops"].append(sum([e.flops for e in prof.key_averages() if e.flops]))
            table_res["total_cpu_time"].append(sum([e.self_cpu_time_total/ 1e6 for e in prof.key_averages()]))
            table_res["total_cuda_time"].append(sum([e.self_cuda_time_total/ 1e6 for e in prof.key_averages()]))
    
        return table_res

def error_eval(model,model_name,config,config_model,obs_points,sol_test,paths, device="cpu"):
    n_resample = int(config.iter_mcmc*0.1)
    n_alpha_blocks = int(config.iter_da*0.1)

    obs_points = torch.tensor(obs_points,device = device, dtype = torch.float64)

    flops_time_res = flops_time(model,obs_points, sol_test,config,device)
        
    gp_use= True if model_name in ["pigp","gp"] else False
    # Individual results
    table_res = {"alpha": None, "m_error": []}

    # for i in range(config.repeat):
    fem_chain_path = os.path.join(config_model.paths.results_dir,f"FEM_mcmc_kl{config.KL_expansion}_{0}.npy")
    fem_eval_path = os.path.join(config_model.paths.results_dir,f"FEM_mcmc_eval_kl{config.KL_expansion}_{0}.npy")
    model_alpha = paths[f"{model_name}_da"].format(i=0)

    chain_fem = torch.tensor(np.load(fem_chain_path),device = device, dtype = torch.float64)
    eval_fem =  torch.tensor(np.load(fem_eval_path),device = device, dtype = torch.float64)
    alpha_model = np.load(model_alpha)
    N = chain_fem.shape[0]

    for s in range(config.repeat):
        seed = 42 + s
        torch.manual_seed(seed)

        indices = torch.randint(low=0, high=N, size=(n_resample,))
        chain_fem_resampled = chain_fem[indices,:]
        eval_fem_resampled = eval_fem[indices,:]
        
        if not config.marginal_approximation:
            error_mcmc = error_norm_mean(model, eval_fem_resampled, obs_points,chain_fem_resampled, device, gp=gp_use)
        else:
            error_mcmc = error_norm_marginal(model,eval_fem_resampled, obs_points,chain_fem_resampled, device, gp=gp_use)
        
        table_res["m_error"].append(error_mcmc)

    stat = stat_ar(alpha_model, every=n_alpha_blocks)[-1]
    table_res["alpha"] = stat.tolist()

    table_res = {**table_res,**flops_time_res}
    
    aprox_type = "marginal" if config.marginal_approximation else "mean"
    if config_model.model_type == "dgala":
        model_id = (
            f"_{aprox_type}"
            f"_hl{config_model.model.num_layers}"
            f"_hd{config_model.model.hidden_dim}"
            f"_s{config_model.nn_samples}"
            f"_b{config_model.batch_size}"
            f"_kl{config_model.KL_expansion}")

    elif config_model.model_type == "pigp":
        model_id = (
            f"_{aprox_type}"
            f"_kernel_spatial{config_model.kernel_spatial}"
            f"_kernel_parameter{config_model.kernel_parameter}"
            f"_spatial{config_model.observed_spatial_points}"
            f"_nsol{config_model.observed_solutions}"
            f"_kl{config_model.KL_expansion}")
        
    elif config_model.model_type == "gp":
        model_id = (
            f"_{aprox_type}"
            f"_kernel{config_model.kernel}"
            f"_spatial{config_model.observed_spatial_points}"
            f"_nsol{config_model.observed_solutions}"
            f"_kl{config_model.KL_expansion}")
        
    saving_path = os.path.join(config_model.paths.results_dir,f"{config_model.model_type}{model_id}.json")
    with open(saving_path, "w") as f:
        json.dump(table_res, f, indent=4)
