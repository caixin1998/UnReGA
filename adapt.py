import argparse
import yaml
import os
import torch
import torch.optim as optim
import random
import time
import numpy as np
import sys
import copy
from torch import nn
from utils import *
import models
from datareader import txtload

def configure_model(model):
    """Configure model for freezing batch_norm."""
    model.eval()
    return model

# test result of statedict on dataset
def test(statedict, dataset, outfile, epoch=0):
    """Test model on dataset.
    
    Args: 
        statedict (dict): Model state_dict.
        dataset (torch.utils.data.Dataloader): Dataset to test on.
        outfile (file): File to write results to.
        epoch (int): Epoch number.
    """
    global best_result_dict
    net = models.GazeRes(args.backbone)
    net.to(device)
    configure_model(net)
    net.load_state_dict(statedict, strict=False)
    accs = 0
    count = 0
    with torch.no_grad():
        for j, (data, label) in enumerate(dataset):
            img = data["face"].to(device)
            names = data["name"]

            img = {"face": img}
            gts = label.to(device)
            gazes, _ = net(img)
            accs += torch_angular_error(gazes, gts) * gts.shape[0]
            count += gts.shape[0]
        avg_acc = accs / count
        loger = f"[{epoch}] Total Num: {count}, avg: {avg_acc:.3f} \n"
        print(loger)
        outfile.write(loger)
        outfile.flush()
    return avg_acc


def test_ensemble(nets, dataset, outfile, epoch=0):
    """
    Test average performance of models(nets) on dataset.
    Args:
        nets (list): List of models to test.
        dataset (torch.utils.data.Dataloader): Dataset to test on.
        outfile (file): File to write results to.
        epoch (int): Epoch number.
    """
    
    for net in nets:
        net.eval()
    accs = 0
    count = 0
    with torch.no_grad():
        for j, (data, label) in enumerate(dataset):
            img = data["face"].to(device)
            names = data["name"]

            img = {"face": img}
            gts = label.to(device)
            avg_gazes = 0
            for net in nets:
                gazes, _ = net(img)
                avg_gazes = avg_gazes + gazes
            avg_gazes = avg_gazes / len(nets)
            accs += torch_angular_error(gazes, gts) * gts.shape[0]
            count += gts.shape[0]

        avg_acc = accs / count
        loger = f"[{epoch}] Total Num: {count}, avg: {avg_acc:.3f} \n"
        print(loger)
        outfile.write(loger)

    return avg_acc

def train_test(train_data, test_data, iteration, adapt_loss_op, outfile):
    """
    Train and test model on dataset.
    Args:
        train_data (torch.utils.data.Dataloader): Dataset to train on.
        test_data (torch.utils.data.Dataloader): Dataset to test on.
        nets (list): List of models to train.
        nets_ema (list): List of EMA models to train.
        iteration (int): Number of iterations to train.
    """

    # Initialize models
    for i in range(len(nets)):
        nets[i].load_state_dict(nets_init[i].state_dict())
        nets_ema[i].load_state_dict(nets_init[i].state_dict())
        configure_model(nets[i])
        configure_model(nets_ema[i])
    # Optimizer
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.95))
    for i in range(iteration):
 
        gazes = torch.Tensor().to(device)
        gazes_ema = torch.Tensor().to(device)
        # Randomly sample 20 indices from training data
        indices = random.sample(range(train_data["face"].shape[0]), 20)
        img = train_data["face"][indices]
        img = {"face": img}
        for k in range(len(nets)):
            gaze, feature = nets[k](img)
            gazes = torch.cat((gazes, gaze.reshape(-1, 2, 1)), 2)
            gaze_ema, feature = nets_ema[k](img)
            gazes_ema = torch.cat((gazes_ema, gaze_ema.reshape(-1, 2, 1)), 2)
       
        outlier_loss = adapt_loss_op(gazes, gazes_ema)
        optimizer.zero_grad()
        outlier_loss.backward()
        optimizer.step()
        for k in range(len(nets)):
            update_ema_params(nets[k], nets_ema[k], 0.99, i)
        # print(outlier_loss.item())
        outfile.write("Outlier_loss: %.4f \n"%(outlier_loss.item()))
        outfile.flush()
    statedict = mean_models_params(nets)
    error = test(statedict, test_data, outfile, i)
    
    
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--backbone', type=str, default='res18', help='backbone')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--num_models', type=float, default=10, help='number of pretrained models(>1)')
    parser.add_argument('--iteration', type=int, default=50, help='iteration for adaptation')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle adaptation dataset')
    parser.add_argument('--target', type=str, default='mpii', help='target dataset,  mpii/edp/capture')
    parser.add_argument('--source', type=str, default='eth', help='source dataset, eth/gaze360')
    parser.add_argument('--savepath', type=str, default="", help='save path for logs and models')
    parser.add_argument('-l', '--loss', default= "uncertain_wpseudo", help="the loss type for adapt")
    parser.add_argument('-lp', '--lamda_pseudo',type=float, default= 0.0001, help="the weight for pseudo loss")
    parser.add_argument('-n', '--num_experiments', default= 100, help="the number of experiments")
    parser.add_argument('--lr', type=float, default=2e-5, help="the learning rate")
    parser.add_argument('--ckpt_path',default="checkpoints/xgaze", help="the path of source model ckpts")
    # use config file
    # ... parse other arguments ...
    args = parser.parse_args()
    
    # Load configuration
    config = yaml.load(open("datapath.yaml"), Loader=yaml.FullLoader)
    
    imagepath_target = config[args.target]["image"]
    labelpath_target = config[args.target]["label"]
    # Set random seed
    seed_everything(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read target data
    if os.path.isdir(labelpath_target):
        folder_target = os.listdir(labelpath_target)
        folder_target.sort()
    else:
        folder_target = [os.path.basename(labelpath_target)]
        labelpath_target = os.path.dirname(labelpath_target)

    labelpath_list = [os.path.join(labelpath_target, j) for j in folder_target]
    dataset_target_for_adaptation = txtload(labelpath_list, imagepath_target, args.batch_size,
                                    shuffle=args.shuffle, num_workers=4, header=True, target=args.target)
    
    dataset_target = txtload(labelpath_list, imagepath_target, 256,
                                    shuffle=False, num_workers=8, header=True, target = args.target)
    # makdir for savepath
    savepath = os.path.join("adapt_logs", args.savepath, f"batch_size={args.batch_size}_iteration={args.iteration}_lr={args.lr}_loss={args.loss}_shuffle={args.shuffle}")
    if args.loss == "uncertain_pseudo" or args.loss == "uncertain_wpseudo":
        savepath = os.path.join("adapt_logs", args.savepath, f"batch_size={args.batch_size}_iteration={args.iteration}_lamda_pseudo={args.lamda_pseudo}_lr={args.lr}_loss={args.loss}_shuffle={args.shuffle}")
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok = True)

    # Model initialization

    params = []
    loc = "cuda:0"
    device = torch.device(loc if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path
    if os.path.isdir(ckpt_path):
        ckpt_list = os.listdir(ckpt_path)
        # sort ckpt_list
        ckpt_list.sort(key=lambda x: int(x.split("=")[1].split(".")[0]),reverse=True)
        pre_models = [os.path.join(ckpt_path, j) for j in ckpt_list]
    elif os.path.isfile(ckpt_path):
        pre_models = [ckpt_path]
    else:
        raise ValueError("No such ckpt path")

    n = len(pre_models)
    n = min(n,args.num_models)

    nets = [models.GazeRes(args.backbone) for _ in range(n)]
    nets_ema = [models.GazeRes(args.backbone) for _ in range(n)]
    nets_init = [models.GazeRes(args.backbone) for _ in range(n)]


    for i in range(n):
        print(pre_models[i])
        pretrain = torch.load(pre_models[i], map_location=loc)
        statedict = pretrain if "state_dict" not in pretrain else pretrain["state_dict"]
        nets[i].to(device)
        nets[i].load_state_dict(statedict)
        nets[i].eval()
        nets_ema[i].to(device)
        nets_ema[i].load_state_dict(statedict)
        nets_ema[i].eval()
        nets_init[i].to(device)
        nets_init[i].load_state_dict(statedict)
        nets_init[i].eval()
        for value in nets[i].parameters():
            if value.requires_grad:
                params += [{'params': [value]}]
        for param in nets_ema[i].parameters():
            param.detach_()



    # Training loop
    errors = AverageMeter()
    std_list = []
    iteration = args.iteration
    adapt_loss_op = build_adaptation_loss(args.loss, args.lamda_pseudo)
    length_target = len(dataset_target_for_adaptation)
    with open(os.path.join(savepath, "train.log"), "w") as outfile:
        with open(os.path.join(savepath, "loss.log"), "w") as lossfile:
            for j, (data, label) in enumerate(dataset_target_for_adaptation):
                if j == 0:
                    statedict = mean_models_params(nets)
                    test(statedict, dataset_target, outfile, 0)
                    outfile.write(" \n")
                if j > args.num_experiments:
                    break
                label = label.to(device)
                for k, v in data.items():
                    if torch.is_tensor(v):
                        data[k] = v.to(device)
                
                gaze_error = train_test(data, dataset_target, iteration, adapt_loss_op, lossfile)
                errors.update(gaze_error.item(), label.size(0))
                std_list += [gaze_error.item()]
                timeend = time.time()
                log = f"[{j}/{length_target}] " \
                    f"batch_size: {args.batch_size} " \
                    f"iteration: {args.iteration} " \
                    f"avg_loss:{errors.avg:.4f} " \
                    f"gaze_loss:{errors.val:.4f} "
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()
                outfile.flush()
            outfile.write("std = %.4f"%(np.std(std_list)) + "\n")