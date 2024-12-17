from __future__ import absolute_import, division, print_function
from models import *
# import numpy as np
import os
import torch
import time
import argparse
device = 'cuda'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def zm(
    model,
    sample,
    original_label,
    clip_max=torch.tensor(1.0).cuda(),
    clip_min=torch.tensor(0.0).cuda(),
    constraint="l2",
    num_iterations=100,
    gamma=1.0,
    target_label=None,
    target_image=None,
    stepsize_search="geometric_progression",
    max_num_evals=1e4,
    init_num_evals=100,
    verbose=True,
):
    # original_label = np.argmax(model.predict(np.expand_dims(sample),axis=1)
    params = {
        "clip_max": clip_max,
        "clip_min": clip_min,
        "shape": sample.shape,  # [batchsize, 3, w, h]
        "original_label": original_label,
        "target_label": target_label,
        "target_image": target_image,
        "constraint": constraint,
        "num_iterations": num_iterations,
        "gamma": gamma,
        "d": int(torch.prod(torch.tensor(sample.shape)[1:])),
        "stepsize_search": stepsize_search,
        "max_num_evals": max_num_evals,
        "init_num_evals": init_num_evals,
        "verbose": verbose,
    }

    # Set binary search threshold.
    if params["constraint"] == "l2":
        params["theta"] = torch.tensor(params["gamma"]) / (
            torch.sqrt(torch.tensor(params["d"])) * params["d"]
        )
    else:
        params["theta"] = params["gamma"] / (params["d"] ** 2)

    params["theta"] = torch.tensor(params["theta"]).cuda()

    # Initialize.
    perturbed = initialize(model, sample, params)  # [bsz, 3, w, h]

    # Project the initialization to the boundary.

    perturbed, dist_post_update = binary_search_batch(sample, perturbed, model, params)
    dist = compute_distance_batch(perturbed, sample, constraint)
    
    for j in torch.arange(params["num_iterations"]):
        # last_perturbed=perturbed.detach().clone()
        # iter_start_time=time.time()
        params["cur_iter"] = j + 1

        # Choose delta.
        delta = select_delta_batch(params, dist_post_update)  # [bsz]

        # Choose number of evaluations.
        num_evals = int(params["init_num_evals"] * torch.sqrt(j + 1))
        num_evals = int(min([num_evals, params["max_num_evals"]]))

        # start_time=time.time()
        # approximate gradient.
        gradf = approximate_gradient_batch(
            model, perturbed, num_evals, delta, params
        )  # [bsz, 3, w, h]

        # print(f'approximating gradient. {time.time()-start_time} seconds used.' ,flush=True)

        if params["constraint"] == "linf":
            update = torch.sign(gradf)
        else:
            update = gradf

        # search step size.
        if params["stepsize_search"] == "geometric_progression":
            # find step size.
            epsilon = geometric_progression_for_stepsize_batch(
                perturbed, update, dist, model, params
            )

            # Update the sample.
            perturbed = clip_image(perturbed + epsilon * update, clip_min, clip_max)

            # start_time=time.time()
            perturbed, dist_post_update = binary_search_batch(
                sample, perturbed, model, params
            )
            # print(f'boundary search. {time.time()-start_time} seconds used.' ,flush=True)
            # if params["cur_iter"]%10 == 0:

            #     torch.save(perturbed-sample,f"saved_per_wanet/per_{params['cur_iter']}_{original_label}_{target_label}.pth")

        dist = compute_distance_batch(perturbed, sample, constraint)
        
        # bsz=perturbed.shape[0]
        # diff=torch.norm(last_perturbed.view(bsz,-1)-perturbed.view(bsz,-1),dim=-1).mean()
        # cur_iter=params['cur_iter']
        # print(f'time for iter {cur_iter}: {time.time()-iter_start_time}',flush=True)
        # print(f'difference: {diff.item()}',flush=True)

        if verbose:
            print(
                f"iteration: {j+1}, {constraint} distance {dist}")

    return dist, perturbed-sample


def decision_function(model, images, params):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    images = clip_image(images, params["clip_min"], params["clip_max"])

    prob = model(images.float())
    # print(prob.shape)
    # print(torch.argmax(prob, dim=1))
    # exit(0)
    if params["target_label"] is None:
        return torch.argmax(prob, dim=1) != params["original_label"]
        # return torch.argmax(prob, dim=1) != 2
    else:
        # return torch.argmax(prob, dim=1) == params["target_label"]
        return torch.argmax(prob, dim=1) == params["target_label"]


def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.


    return torch.clamp(image, clip_min, clip_max)
    # return


def compute_distance_batch(x_ori, x_pert, constraint="l2"):
    bsz = x_ori.shape[0]
    if constraint == "l2":
        return torch.norm((x_ori - x_pert).reshape(bsz, -1), dim=-1)
    elif constraint == "linf":
        return torch.max(torch.abs(x_ori - x_pert).reshape(bsz, -1), dim=-1)[0]


def approximate_gradient(model, sample, num_evals, delta, params):
    clip_max, clip_min = params["clip_max"], params["clip_min"]

    # Generate random vectors.
    noise_shape = [num_evals] + list(params["shape"])  # [num_evals, bsz, 3, w, h]
    if params["constraint"] == "l2":
        rv = torch.randn(*noise_shape)
    elif params["constraint"] == "linf":
        rv = 2 * torch.rand(noise_shape) - 1

    rv = rv.cuda()
    rv = rv / torch.sqrt(torch.sum(rv**2, dim=(1, 2, 3), keepdim=True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(model, perturbed, params)
    decision_shape = [len(decisions)] + [1] * len(params["shape"])
    fval = 2 * decisions.float().reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if torch.mean(fval) == 1.0:  # label changes.
        gradf = torch.mean(rv, dim=0)
    elif torch.mean(fval) == -1.0:  # label not change.
        gradf = -torch.mean(rv, dim=0)
    else:
        fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, dim=0)

    # Get the gradient direction.
    gradf = gradf / torch.norm(gradf)

    return gradf


def approximate_gradient_batch(model, sample, num_evals, delta, params):
    clip_max, clip_min = params["clip_max"].unsqueeze(dim=0), params[
        "clip_min"
    ].unsqueeze(dim=0)
    bsz = sample.shape[0]
    # Generate random vectors.
    noise_shape = [num_evals] + list(params["shape"])  # [num_evals, bsz, 3, w, h]
    if params["constraint"] == "l2":
        rv = torch.randn(*noise_shape)
    elif params["constraint"] == "linf":
        rv = 2 * torch.rand(noise_shape) - 1

    rv = rv.cuda()
    rv = rv / torch.sqrt(torch.sum(rv**2, dim=(2, 3, 4), keepdim=True))

    sample = sample.unsqueeze(dim=0)
    delta = delta.reshape(1, -1, 1, 1, 1)
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)

    # if params['cur_iter'] % 10==0:

    #     print("+++++++++++++++++saved Pers+++++++++++++++++++++++")
    #     torch.save(perturbed,f"saved_per_tsne/per_wanet_{params['cur_iter']}.pth")
    rv = (perturbed - sample) / delta

    tot_grad = []
    for idx in range(bsz):
        decisions = decision_function(model, perturbed[:, idx, :, :, :], params)
        decision_shape = [len(decisions)] + [1] * (len(params["shape"]) - 1)
        fval = 2 * decisions.float().reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if torch.mean(fval) == 1.0:  # label changes.
            gradf = torch.mean(rv[:, idx, :, :, :], dim=0)
        elif torch.mean(fval) == -1.0:  # label not change.
            gradf = -torch.mean(rv[:, idx, :, :, :], dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv[:, idx, :, :, :], dim=0)
        gradf = gradf / torch.norm(gradf)
        tot_grad.append(gradf)
    tot_grad = torch.stack(tot_grad, dim=0)

    return tot_grad


def project_batch(original_image, perturbed_images, alphas, params):
    alphas_shape = [len(alphas)] + [1] * (len(params["shape"]) - 1)
    alphas = alphas.reshape(alphas_shape)
    if params["constraint"] == "l2":
        return (1 - alphas) * original_image + alphas * perturbed_images
    elif params["constraint"] == "linf":
        out_images = clip_image(
            perturbed_images, original_image - alphas, original_image + alphas
        )
        return out_images


def binary_search_batch(original_image, perturbed_images, model, params):
    """Binary search to approach the boundar."""

    dists_post_update = compute_distance_batch(
        original_image, perturbed_images, params["constraint"]
    )

    # print(original_image.shape)
    # print(perturbed_images.shape)
    # print(dists_post_update.shape)

    # Choose upper thresholds in binary searchs based on constraint.
    if params["constraint"] == "linf":
        highs = dists_post_update
        # Stopping criteria.
        thresholds = torch.minimum(dists_post_update * params["theta"], params["theta"])
    else:
        highs = torch.ones(perturbed_images.shape[0]).cuda()
        thresholds = params["theta"]

    lows = torch.zeros(perturbed_images.shape[0]).cuda()

    # Call recursive function.
    while torch.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        # print(torch.max((highs - lows) / thresholds),flush=True)
        mids = (highs + lows) / 2.0

        mid_images = project_batch(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.

        decisions = decision_function(model, mid_images, params)

        lows = torch.where(decisions == 0, mids, lows)
        highs = torch.where(decisions == 1, mids, highs)

    out_images = project_batch(original_image, perturbed_images, highs, params)
    dists = compute_distance_batch(original_image, out_images, params["constraint"])

    return out_images, dists


"""used when stepsize_search is grid_search  can be problematic here!"""
# out_images = project(original_image, perturbed_images, highs, params)

# # Compute distance of the output image to select the best choice.
# # (only used when stepsize_search is grid_search.)
# dists = torch.stack(
#     [
#         compute_distance(original_image, out_image, params["constraint"])
#         for out_image in out_images
#     ],
#     dim=0,
# )
# idx = torch.argmin(dists)

# dist = dists_post_update[idx]
# out_image = out_images[idx]
# return out_image, dist


def initialize(model, sample, params):
    """
    Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    """
    success = 0
    num_evals = 0

    if params["target_image"] is None:
        # Find a misclassified random noise.
        random_noise = []
        bsz = sample.shape[0]
        tot_num = 0

        while True:
            new_random_noise = torch.rand_like(sample).cuda()
            new_random_noise = (
                new_random_noise * (params["clip_max"] - params["clip_min"])
                + params["clip_min"]
            )

            cur_success = decision_function(model, new_random_noise, params)

            tot_num += torch.sum(cur_success)

            # print(tot_num)

            random_noise.append(new_random_noise[cur_success])

            num_evals += 1
            if tot_num >= bsz:
                break
            assert num_evals < 1e4, "Initialization failed! "
            "Use a misclassified image as `target_image`"
        random_noise = torch.cat(random_noise, dim=0)
        random_noise = random_noise[:bsz]
        # print('random noise shape:',random_noise.shape)
        # success=decision_function(model,random_noise,params)
        # print(torch.sum(success))
        # exit(0)

        # Binary search to minimize l2 distance to original image.

        low = torch.zeros(bsz).cuda()
        high = torch.ones(bsz).cuda()
        while torch.max(high - low) > 0.001:
            mid = (high + low) / 2.0

            blended = (1 - mid.view(-1, 1, 1, 1)) * sample + mid.view(
                -1, 1, 1, 1
            ) * random_noise
            success = decision_function(model, blended, params)

            low = torch.where(success == 0, mid, low)
            high = torch.where(success == 1, mid, high)

        initialization = (1 - high.view(-1,1,1,1)) * sample + high.view(-1,1,1,1) * random_noise

    else:
        initialization = params["target_image"]

    return initialization


def geometric_progression_for_stepsize_batch(x, update, dist, model, params):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching
    the desired side of the boundary,
    """
    epsilon = (dist / torch.sqrt(params["cur_iter"])).reshape(-1, 1, 1, 1)  # [bsz]

    def phi(epsilon, mask):
        new = x[mask] + epsilon[mask] * update[mask]
        success = decision_function(model, new, params)
        return success

    mask = torch.ones(epsilon.shape[0]).bool().cuda()  # to run
    success = torch.zeros(epsilon.shape[0]).bool().cuda()

    stop_cnt = 0  # add by crb to avoid being stuck
    while True:
        stop_cnt += 1
        success[mask] = phi(epsilon, mask)  # [bsz]
        mask = success.logical_not()

        if torch.sum(success) == success.shape[0] or (stop_cnt >= 25):
            break

        epsilon = torch.where(success.reshape(-1, 1, 1, 1), epsilon, epsilon / 2)

    return epsilon


def select_delta_batch(params, dist_post_update):
    """
    Choose the delta at the scale of distance
    between x and perturbed sample.

    """
    if params["cur_iter"] == 1:
        delta = 0.1 * torch.min(params["clip_max"] - params["clip_min"]).reshape(
            1
        ).expand(dist_post_update.shape[0])
    else:
        if params["constraint"] == "l2":
            delta = (
                torch.sqrt(torch.tensor(params["d"]))
                * params["theta"]
                * dist_post_update
            )
        elif params["constraint"] == "linf":
            delta = params["d"] * params["theta"] * dist_post_update

    return delta.cuda()

if __name__ == "__main__": 
        
        parser = argparse.ArgumentParser()



        parser.add_argument('--ori', type=int)

        parser.add_argument('--target', type=int)

     
        args = parser.parse_args()

        if True:

            original_label = args.ori
            #original_label = h

        # strore the tensor for training data by yourself
            sample = torch.load(f"data_tensor/cifar_{original_label}.pth")[:200]
        

            target_label = args.target
            target_label = 0
            target_imgs = torch.load(f"data_tensor/cifar_{target_label}.pth")[:200]

            model = VGG('VGG19')
            model = torch.nn.DataParallel(model)
        
        #cudnn.benchmark = True
        
            model = model.to(device)
        
            checkpoint = torch.load('./saved_models/ckpt_blended_over.pth')
        
            model.load_state_dict(checkpoint['net'])
    
            model.eval()

            sample = sample.to(device)
        
            target_imgs = target_imgs.to(device)
            
            print(f"{original_label} ->>>>>>>> {target_label}")
            
            dist, perturbed = zm(model,sample=sample, original_label = original_label, target_image=target_imgs, target_label=target_label)

            torch.save(perturbed.detach(),f"saved_per_vgg/per_{original_label}_{target_label}.pth")
