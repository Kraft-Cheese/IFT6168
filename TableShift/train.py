"""
Train EQRM and baselines on TableShift tabular distribution shift benchmarks.

Usage:
    # EQRM on diabetes readmission
    python train.py --task diabetes_readmission --algorithm eqrm \
        --alpha 0.75 --erm_pretrain_iters 500 --steps 2000

    # ERM baseline
    python train.py --task diabetes_readmission --algorithm erm --steps 2000

    # Sweep over all quick tasks
    for task in diabetes_readmission anes food_stamps; do
        for algo in erm eqrm vrex groupdro; do
            python train.py --task $task --algorithm $algo --seed 0
        done
    done
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import json
import time
import copy
import os
import hashlib
import sys
import random
import math

from tqdm import tqdm
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from lib import misc
import algorithms
import networks
from tableshift_datasets import get_tableshift_datasets, DG_TASKS, QUICK_TASKS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EQRM on TableShift')

    # Task
    parser.add_argument('--task', type=str, default='diabetes_readmission',
                        choices=list(DG_TASKS.keys()))
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='TableShift cache directory')

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='eqrm')
    parser.add_argument('--penalty_weight', type=float, default=1000)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--groupdro_eta', type=float, default=1.)

    # Network
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    # Training
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_factor_reduction', type=float, default=10)
    parser.add_argument('--lr_cos_sched', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--erm_pretrain_iters', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=100)

    # Directories
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--exp_name', type=str, default='tableshift_eqrm')
    parser.add_argument('--save_ckpts', action='store_true')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--n_workers', type=int, default=0)

    args = parser.parse_args()
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    loss_fn = F.cross_entropy

    # -------- LOGGING --------
    logs_dir = os.path.join(args.output_dir, "logs", args.exp_name)
    results_dir = os.path.join(args.output_dir, "results", args.exp_name)
    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    sys.stdout = misc.Tee(os.path.join(logs_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(logs_dir, 'err.txt'))
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    # -------- REPRODUCIBILITY --------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- DATA LOADING --------
    print(f"\nLoading TableShift task: {args.task}")
    print(f"  Shift type: {DG_TASKS[args.task]['shift']}")
    train_envs, test_envs, env_names, input_dim = get_tableshift_datasets(
        args.task, cache_dir=args.cache_dir
    )

    # Move to device
    if device == "cuda":
        train_envs = [
            TensorDataset(x.cuda(), y.cuda())
            for x, y in [env.tensors for env in train_envs]
        ]

    # Create data loaders
    train_loaders = [
        InfiniteDataLoader(dataset=env, batch_size=args.batch_size, num_workers=args.n_workers)
        for env in train_envs
    ]
    train_minibatches_iterator = zip(*train_loaders)

    test_loaders = [
        FastDataLoader(dataset=env, batch_size=512, num_workers=args.n_workers)
        for env in test_envs
    ]

    # -------- NETWORK --------
    net = networks.TabularMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout_p,
        num_classes=2,  # all TableShift tasks are binary
    )
    print(f"\nNetwork: TabularMLP ({sum(p.numel() for p in net.parameters()):,} params)")
    print(f"  input_dim={input_dim}, hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}")

    # -------- ALGORITHM --------
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(net, vars(args), loss_fn)
    algorithm.to(device)
    print(f"Algorithm: {args.algorithm.upper()}")

    # -------- LOAD ERM CHECKPOINT --------
    start_step = 1
    alg_arg_keys = ["algorithm", "penalty_weight", "alpha", "groupdro_eta",
                    "lr_factor_reduction", "lr_cos_sched", "steps", "save_ckpts"]
    if args.erm_pretrain_iters > 0:
        erm_args = vars(copy.deepcopy(args))
        for k in alg_arg_keys:
            if k in erm_args:
                del erm_args[k]
        erm_ckpt_name = hashlib.md5(str(erm_args).encode('utf-8')).hexdigest()
        erm_ckpt_pth = os.path.join(ckpt_dir, f"{erm_ckpt_name}.pkl")
        if os.path.exists(erm_ckpt_pth):
            algorithm.load_state_dict(torch.load(erm_ckpt_pth, map_location=device), strict=False)
            print(f"ERM-pretrained model loaded.")
            start_step = args.erm_pretrain_iters + 1

    # -------- LR SCHEDULING --------
    def adjust_learning_rate(optimizer, current_step, lr, total_steps):
        lr_adj = lr * 0.5 * (1. + math.cos(math.pi * current_step / total_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adj

    # -------- TRAINING LOOP --------
    from torch.utils.data import TensorDataset

    results = {}
    best_metric, best_weights = -1., copy.deepcopy(algorithm.state_dict())

    pbar = tqdm(range(start_step, args.steps + 1), desc="Training", unit="step")
    for step in pbar:
        # LR schedule
        if args.lr_cos_sched and args.algorithm.lower() != "erm":
            if args.erm_pretrain_iters == 0:
                adjust_learning_rate(algorithm.optimizer, step, args.lr, args.steps)
            elif step > args.erm_pretrain_iters > 0:
                lr_ = args.lr / args.lr_factor_reduction
                steps_ = args.steps - args.erm_pretrain_iters
                step_ = step - args.erm_pretrain_iters
                adjust_learning_rate(algorithm.optimizer, step_, lr_, steps_)

        # Update
        phase = "ERM pretrain" if step <= args.erm_pretrain_iters else args.algorithm.upper()
        minibatches = next(train_minibatches_iterator)
        minibatches = [(x.to(device), y.to(device)) for x, y in minibatches]
        step_values = algorithm.update(minibatches)
        pbar.set_postfix(loss=f"{step_values.get('loss', 0):.4f}", phase=phase)

        # Evaluation
        if step % args.eval_freq == 0 or step == args.steps:
            results.update({'step': step})
            for key, val in step_values.items():
                results[key] = val

            for env_name, env_loader in zip(env_names['test'], test_loaders):
                acc = misc.accuracy(algorithm, env_loader, device)
                results[f'{env_name}_acc'] = acc

            ood_acc = results.get('OOD_test_acc', 0)
            id_acc = results.get('ID_val_acc', 0)
            gap = id_acc - ood_acc

            pbar.set_postfix(
                loss=f"{step_values.get('loss', 0):.4f}",
                ood=f"{ood_acc:.4f}",
                gap=f"{gap:.4f}",
                phase=phase,
            )
            tqdm.write(
                f"  [Step {step:5d}] loss={step_values.get('loss', 0):.4f} | "
                f"OOD={ood_acc:.4f} | ID={id_acc:.4f} | gap={gap:.4f}"
            )

            if ood_acc > best_metric:
                best_metric = ood_acc
                best_weights = copy.deepcopy(algorithm.state_dict())

        # Save ERM checkpoint
        if step == args.erm_pretrain_iters > 0 and args.save_ckpts:
            torch.save(algorithm.state_dict(), erm_ckpt_pth)
            tqdm.write("Saved ERM-pretrained model.")

    # -------- FINAL EVALUATION --------
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}\n")

    final_results = {}
    for ms_name, weights in [("final", algorithm.state_dict()), ("best", best_weights)]:
        algorithm.load_state_dict(weights)

        for env_name, env_loader in zip(env_names['test'], test_loaders):
            acc = misc.accuracy(algorithm, env_loader, device)
            final_results[f'{env_name}_acc_{ms_name}'] = acc

        ood = final_results.get(f'OOD_test_acc_{ms_name}', 0)
        id_v = final_results.get(f'ID_val_acc_{ms_name}', 0)
        gap = id_v - ood

        print(f"{ms_name.upper()} model:")
        print(f"  OOD test accuracy: {ood:.4f}")
        print(f"  ID val accuracy:   {id_v:.4f}")
        print(f"  Shift gap:         {gap:.4f}")

    # -------- SAVE --------
    args_no_seed = copy.deepcopy(args)
    delattr(args_no_seed, "seed")
    args_id = hashlib.md5(str(args_no_seed).encode('utf-8')).hexdigest()

    final_results["algorithm"] = args.algorithm.lower()
    final_results["task"] = args.task
    final_results["shift_type"] = DG_TASKS[args.task]["shift"]
    final_results["num_train_envs"] = len(train_envs)
    final_results["seed"] = args.seed
    final_results["args_id"] = args_id
    final_results["args"] = vars(args_no_seed)

    results_path = os.path.join(results_dir, f"{md5_fname}.jsonl")
    with open(results_path, 'a') as f:
        f.write(json.dumps(final_results, sort_keys=True, default=str) + "\n")
    print(f"\nResults saved to {results_path}")
