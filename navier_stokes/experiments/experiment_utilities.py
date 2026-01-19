import os
import sys
import json  # or import pickle if you prefer pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "navier_stokes"))  # Explicitly add Elliptic folder

from Base.lla import dgala
from Base.utilities import *
from Base.gp_base import MultioutputGP

from nv_files.nv_mcmc import NVMCMCDA
from nv_files.train_nvs import train_vorticity_epoch
from nv_files.utilities import generate_noisy_obs,deepgala_data_fit
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
            f"_s{config.points_per_chunk}"
            f"_b{config.batch_size}"
            f"_kl{config.KL_expansion}")

        # NN paths
        paths = {
            "dgala_mean_model": os.path.join(config.paths.model_dir, f"dgala_nv_mean{model_id}.pth"),
            "dgala_marginal_model": os.path.join(config.paths.model_dir, f"dgala_nv_marginal{model_id}.pth"),
            "dgala_mcmc": os.path.join(config.paths.results_dir, f"dgala_nv_mcmc_{aprox_type}{model_id}" + "_{i}.npy"),
            "dgala_da": os.path.join(config.paths.results_dir, f"dgala_nv_da_mcmc_{aprox_type}{model_id}" + "_{i}.npy"),
            "dgala_da_chain": os.path.join(config.paths.results_dir, f"dgala_nv_da_chain_{aprox_type}{model_id}" + "_{i}.npy"),
            "dgala_mean_proposal": os.path.join(config.paths.results_dir, f"dgala_nv_mean_proposal_{model_id}.npy"),

            }

        return paths
    
    elif model_type == "gp":
        aprox_type = "marginal" if config.marginal else "mean"
        model_id = (
            f"_kernel{config.kernel}"
            f"_spatial{config.observed_spatial_points}"
            f"_nsol{config.observed_solutions}"
            f"_kl{config.KL_expansion}")
        
        path_gp_model = os.path.join(config.paths.model_dir,f"gp2d{model_id}.pth")
        path_gp_mcmc = os.path.join(config.paths.results_dir,f"gp2d_mcmc_{aprox_type}{model_id}" + "_{i}.npy")
        path_gp_damcmc = os.path.join(config.paths.results_dir,f"gp2d_da_mcmc_{aprox_type}{model_id}" + "_{i}.npy")

        return {"gp_model": path_gp_model,"gp_mcmc":path_gp_mcmc,"gp_da_mcmc":path_gp_damcmc}
    
    elif model_type == "psm":
        model_id = (f"_kl{config.KL_expansion}")
        
        fem_mcmc = os.path.join(config.paths.results_dir,f"psm_mcmc{model_id}" + "_{i}.npy")
        fem_mcmc_eval = os.path.join(config.paths.results_dir,f"psm_mcmc_eval{model_id}" + "_{i}.npy")

        return {"psm_mcmc": fem_mcmc,"psm_mcmc_eval":fem_mcmc_eval}
    
def train_neural_network(config_experiment,path,device):
    model_specific = f"_nv_mean_hl{config_experiment.model.num_layers}_hd{config_experiment.model.hidden_dim}_s{config_experiment.points_per_chunk}_b{config_experiment.batch_size}_kl{config_experiment.KL_expansion}"

    print(f"Running training with {config_experiment.points_per_chunk} samples...")
    config_experiment.wandb.name = config_experiment.model_type + model_specific
    config_experiment.batch_ic = 16*config_experiment.points_per_chunk
    
    config_experiment.model.input_dim = 3 + 2*config_experiment.KL_expansion
    config_experiment.model.fourier_emb["exclude_last_n"] = 2*config_experiment.KL_expansion
    
    pinn_nvs = train_vorticity_epoch(config_experiment, device=device)

    print(f"Completed training model {config_experiment.model_type}{model_specific}.")

    if config_experiment.dgala:
        nn_model = torch.load(path["dgala_mean_model"], map_location=device)
        nn_model.eval()
        data_fit = deepgala_data_fit(config_experiment,device)
        llp = dgala(nn_model)
        llp.fit(data_fit)
        llp.optimize_marginal_likelihood()
        clear_hooks(llp)
        torch.save(llp, path["dgala_marginal_model"])


def fit_deepgala(config_experiment,path,device):
    print(f"Starting DeepGaLA fitting for NN_s{16*config_experiment.points_per_chunk}")

    model_path = path["dgala_mean_model"]
    if os.path.exists(model_path):
            nn_model = torch.load(model_path, map_location=device)
            nn_model.eval()
    else: 
        train_neural_network(config_experiment,path,device)
        nn_model = torch.load(model_path, map_location=device)
        nn_model.eval()

    data_fit = deepgala_data_fit(config_experiment,device)
    llp = dgala(nn_model)
    llp.fit(data_fit)
    llp.optimize_marginal_likelihood()
    clear_hooks(llp)
    torch.save(llp, path["dgala_marginal_model"])


# def train_gp(config_experiment,path,device):    
#     print(f"Training GP_s{config_experiment.observed_solutions}")
#     data_training,_,_ = pigp_training_data_generation(config_experiment.observed_solutions,config_experiment.observed_spatial_points,config_experiment.KL_expansion,device)
#     print(data_training)
#     mine_gp = MultioutputGP(data_training,kernel =config_experiment.kernel)
#     mine_gp.train_gp()
#     mine_gp.optimize_gp()

#     torch.save(mine_gp, path["gp_model"])


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

    # elif config.model_type == "gp":
    #     model_path = paths.get("gp_model")
    #     if not os.path.exists(model_path):
    #         train_gp(config_model,paths,device)
    #     models[f"gp"] = torch.load(paths["gp_model"], map_location=device)

    return model_name,model

############
def get_sampler(model,config_experiment,device):
     # Step 3: Generate noisy observations for Inverse Problem
    obs_points, sol_test, obs_indices,_  = generate_noisy_obs(obs=config_experiment.num_observations,
                                               noise_level=config_experiment.noise_level,
                                               NKL = config_experiment.KL_expansion,
                                               seed = config_experiment.seed)
    
    print(_)
    

    elliptic_mcmcda = NVMCMCDA(model, 
                        observation_locations= obs_points,fs_indices_sol = obs_indices,
                          observations_values = sol_test, 
                        nparameters=2*config_experiment.KL_expansion,
                        marginal = config_experiment.marginal_approximation,
                        observation_noise=np.sqrt(config_experiment.noise_level),
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        proposal_type=config_experiment.proposal,
                        prior_type=config_experiment.prior_type,
                        uniform_limit = config_experiment.uniform_limit,
                        step_size=config_experiment.proposal_variance, 
                        device=device) 
    return elliptic_mcmcda,obs_points, sol_test


# def flops_time(model,obs_points, sol_test,config_experiment,device):
#         samples = samples_param(10_000, config_experiment.KL_expansion)
        
#         obs_points = torch.tensor(obs_points,device = device, dtype = torch.float64)

#         mcmc = NVMCMCDA(model,None, obs_points, sol_test, 
#                               nparameters=config_experiment.KL_expansion,marginal=config_experiment.marginal_approximation,
#                               observation_noise=np.sqrt(config_experiment.noise_level),device=device)

#         table_res = {"total_cpu_time":[],"total_cuda_time":[],"total_flops":[]}
#         for sample in samples:
#             theta = torch.tensor(sample,device = device, dtype = torch.float64).reshape(-1)

#             with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
#                 mcmc.log_likelihood_outer(theta)
                                
#             table_res["total_flops"].append(sum([e.flops for e in prof.key_averages() if e.flops]))
#             table_res["total_cpu_time"].append(sum([e.self_cpu_time_total/ 1e6 for e in prof.key_averages()]))
#             table_res["total_cuda_time"].append(sum([e.self_cuda_time_total/ 1e6 for e in prof.key_averages()]))
    
#         return table_res

def error_eval(model,model_name,config,config_model,obs_points,sol_test,paths, device="cpu"):
    n_resample = int(config.iter_mcmc*0.1)
    n_alpha_blocks = int(config.iter_da*0.1)

    obs_points = torch.tensor(obs_points,device = device, dtype = torch.float64)

    #flops_time_res = flops_time(model,obs_points, sol_test,config,device)
        
    gp_use= True if model_name in ["pigp","gp"] else False
    # Individual results
    table_res = {"alpha": None, "m_error": []}

    # for i in range(config.repeat):
    # fem_chain_path = os.path.join(config_model.paths.results_dir,f"FEM2d_mcmc_kl{config.KL_expansion}_{0}.npy")
    # fem_eval_path = os.path.join(config_model.paths.results_dir,f"FEM2d_mcmc_eval_kl{config.KL_expansion}_{0}.npy")
    model_alpha = paths[f"{model_name}_da"].format(i=0)

    # chain_fem = torch.tensor(np.load(fem_chain_path),device = device, dtype = torch.float64)
    # eval_fem =  torch.tensor(np.load(fem_eval_path),device = device, dtype = torch.float64)
    alpha_model = np.load(model_alpha)
    # N = chain_fem.shape[0]

    # for s in range(config.repeat):
    #     seed = 42 + s
    #     torch.manual_seed(seed)
    #     indices = torch.randint(low=0, high=N, size=(n_resample,))
    #     chain_fem_resampled = chain_fem[indices,:]
    #     eval_fem_resampled = eval_fem[indices,:]
        
    #     if not config.marginal_approximation:
    #         error_mcmc = error_norm_mean(model, eval_fem_resampled, obs_points,chain_fem_resampled, device, gp=gp_use)
    #     else:
    #         error_mcmc = error_norm_marginal(model,eval_fem_resampled, obs_points,chain_fem_resampled, device, gp=gp_use)
        
    #     table_res["m_error"].append(error_mcmc)

    stat = stat_ar(alpha_model, every=n_alpha_blocks)[-1]
    table_res["alpha"] = stat.tolist()
    #table_res["alpha"].append(alpha_model.sum() / alpha_model.shape[-1])

    #table_res = {**table_res,**flops_time_res}
    
    aprox_type = "marginal" if config.marginal_approximation else "mean"
    if config_model.model_type == "dgala":
        model_id = (
            f"_{aprox_type}"
            f"_hl{config_model.model.num_layers}"
            f"_hd{config_model.model.hidden_dim}"
            f"_s{config_model.points_per_chunk}"
            f"_b{config_model.batch_size}"
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