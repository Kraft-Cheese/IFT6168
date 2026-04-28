"""
Train EQRM and baselines on Wild-Time temporal distribution shift benchmarks.

Usage:
    # EQRM on Yearbook with 8 time-window environments
    python train.py --dataset yearbook --algorithm eqrm --num_train_envs 8 \
        --alpha -500 --erm_pretrain_iters 500 --steps 2000

    # ERM baseline
    python train.py --dataset yearbook --algorithm erm --steps 2000

    # VREx
    python train.py --dataset yearbook --algorithm vrex --num_train_envs 8 \
        --penalty_weight 1000 --erm_pretrain_iters 500 --steps 2000
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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
import algorithms
import networks
from datasets import get_datasets, DATASET_REGISTRY


if __name__ == "__main__":
    # -------- FLAGS --------
    parser = argparse.ArgumentParser(description='EQRM on Wild-Time')

    # Dataset
    parser.add_argument('--dataset', type=str, default='yearbook',
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument('--split_time', type=int, default=None,
                        help='Train/test split timestamp (dataset default if None)')
    parser.add_argument('--num_train_envs', type=int, default=8,
                        help='Number of training time-window environments')

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='eqrm')
    parser.add_argument('--penalty_weight', type=float, default=1000)
    parser.add_argument('--alpha', type=float, default=-500,
                        help='EQRM alpha. Negative = log(1-alpha) for values close to 1')
    parser.add_argument('--groupdro_eta', type=float, default=1.)

    # General hparams
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_factor_reduction', type=float, default=10)
    parser.add_argument('--lr_cos_sched', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--erm_pretrain_iters', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=100)

    # Directories
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--exp_name', type=str, default='wildtime_eqrm')
    parser.add_argument('--save_ckpts', action='store_true')

    # W&B
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='IFT6168-WildTime')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Reproducibility
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--n_workers', type=int, default=0)

    # -------- SETUP --------
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

    # W&B init
    use_wandb = args.use_wandb and _WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.algorithm}_{args.dataset}_env{args.num_train_envs}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

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

    # BUGFIX: missing MPS branch
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # -------- DATA LOADING --------
    print(f"\nLoading {args.dataset} with {args.num_train_envs} training environments...")
    train_envs, test_envs, env_names = get_datasets(
        args.dataset, args.data_dir,
        split_time=args.split_time,
        num_train_envs=args.num_train_envs
    )
    print(f"  Train envs: {env_names['train']} ({len(train_envs)} envs)")
    print(f"  Test envs:  {env_names['test'][:5]}... ({len(test_envs)} envs)")
    for i, env in enumerate(train_envs):
        print(f"  Train env {env_names['train'][i]}: {len(env)} samples")

    # Move to device (only for TensorDatasets that fit in GPU memory)
    if device == "cuda" and hasattr(train_envs[0], 'tensors'):
        train_envs_cuda = []
        for env in train_envs:
            x, y = env.tensors
            train_envs_cuda.append(torch.utils.data.TensorDataset(x.cuda(), y.cuda()))
        train_envs = train_envs_cuda

    # Create data loaders (one per environment, zipped together)
    train_loaders = [
        InfiniteDataLoader(dataset=env, batch_size=args.batch_size, num_workers=args.n_workers)
        for env in train_envs
    ]
    train_minibatches_iterator = zip(*train_loaders)

    # Test loaders — subsample to speed up eval if many test years
    test_env_names = env_names['test']
    test_loaders = [
        FastDataLoader(dataset=env, batch_size=256, num_workers=args.n_workers)
        for env in test_envs
    ]

    # -------- NETWORK --------
    ds_info = DATASET_REGISTRY[args.dataset]
    net = networks.get_network(ds_info["network"], num_classes=ds_info["num_classes"])

    # -------- ALGORITHM --------
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(net, vars(args), loss_fn)
    algorithm.to(device)
    print(f"\nAlgorithm: {args.algorithm.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in algorithm.parameters()):,}")

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
            print(f"ERM-pretrained model loaded: {erm_ckpt_name}.")
            start_step = args.erm_pretrain_iters + 1

    # -------- LR SCHEDULING --------
    def adjust_learning_rate(optimizer, current_step, lr, total_steps):
        lr_adj = lr * 0.5 * (1. + math.cos(math.pi * current_step / total_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adj

    # -------- TRAINING LOOP --------
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
        # Move to device if not already there (needed for lazy-loading datasets like FMoW)
        minibatches = [(x.to(device), y.to(device)) for x, y in minibatches]
        step_values = algorithm.update(minibatches)
        pbar.set_postfix(loss=f"{step_values.get('loss', 0):.4f}", phase=phase)

        # Evaluation
        if step % args.eval_freq == 0 or step == args.steps:
            results.update({'step': step})
            for key, val in step_values.items():
                results[key] = val

            # Evaluate on all test environments
            test_metrics = []
            for env_name, env_loader in zip(test_env_names, test_loaders):
                acc = misc.accuracy(algorithm, env_loader, device)
                results[f'test_{env_name}_acc'] = acc
                test_metrics.append(acc)

            avg_ood = np.mean(test_metrics)
            worst_ood = np.min(test_metrics)
            results['avg_ood_acc'] = avg_ood
            results['worst_ood_acc'] = worst_ood

            # Quantile metrics (the EQRM paper's evaluation protocol)
            sorted_metrics = np.sort(test_metrics)
            for q_level in [0.1, 0.25, 0.5, 0.75, 0.9]:
                idx = min(int(q_level * len(sorted_metrics)), len(sorted_metrics) - 1)
                results[f'quantile_{q_level}_acc'] = sorted_metrics[idx]

            pbar.set_postfix(
                loss=f"{step_values.get('loss', 0):.4f}",
                avg_ood=f"{avg_ood:.4f}",
                worst=f"{worst_ood:.4f}",
                phase=phase,
            )
            tqdm.write(
                f"  [Eval step {step:5d}] loss={results.get('loss', 0):.4f} | "
                f"avg_ood={avg_ood:.4f} | worst_ood={worst_ood:.4f}"
            )

            if use_wandb:
                wandb.log({
                    'step': step,
                    'loss': results.get('loss', 0),
                    'avg_ood_acc': avg_ood,
                    'worst_ood_acc': worst_ood,
                    **{k: v for k, v in results.items() if k.startswith('quantile_')},
                })

            if avg_ood > best_metric:
                best_metric = avg_ood
                best_weights = copy.deepcopy(algorithm.state_dict())

        # Save ERM checkpoint
        if step == args.erm_pretrain_iters > 0 and args.save_ckpts:
            torch.save(algorithm.state_dict(), erm_ckpt_pth)
            tqdm.write("Saved ERM-pretrained model.")

    # -------- FINAL EVALUATION --------
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}\n")

    final_results = {}
    for ms_name, weights in [("final", algorithm.state_dict()), ("best", best_weights)]:
        algorithm.load_state_dict(weights)

        all_metrics = []
        for env_name, env_loader in zip(test_env_names, test_loaders):
            acc = misc.accuracy(algorithm, env_loader, device)
            final_results[f'{env_name}_acc_{ms_name}'] = acc
            all_metrics.append(acc)

        avg = np.mean(all_metrics)
        worst = np.min(all_metrics)
        final_results[f'avg_ood_acc_{ms_name}'] = avg
        final_results[f'worst_ood_acc_{ms_name}'] = worst

        # Quantile evaluation
        sorted_m = np.sort(all_metrics)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            idx = min(int(q * len(sorted_m)), len(sorted_m) - 1)
            final_results[f'q{q}_acc_{ms_name}'] = sorted_m[idx]

        print(f"\n{ms_name.upper()} model results:")
        print(f"  Average OOD accuracy:  {avg:.4f}")
        print(f"  Worst OOD accuracy:    {worst:.4f}")
        print(f"  Quantile 0.10:         {final_results[f'q0.1_acc_{ms_name}']:.4f}")
        print(f"  Quantile 0.25:         {final_results[f'q0.25_acc_{ms_name}']:.4f}")
        print(f"  Quantile 0.50 (median):{final_results[f'q0.5_acc_{ms_name}']:.4f}")
        print(f"  Quantile 0.75:         {final_results[f'q0.75_acc_{ms_name}']:.4f}")
        print(f"  Quantile 0.90:         {final_results[f'q0.9_acc_{ms_name}']:.4f}")
        print(f"  Per-env: {[f'{a:.3f}' for a in all_metrics]}")

    # -------- SAVE --------
    args_no_seed = copy.deepcopy(args)
    delattr(args_no_seed, "seed")
    args_id = hashlib.md5(str(args_no_seed).encode('utf-8')).hexdigest()

    final_results["algorithm"] = args.algorithm.lower()
    final_results["dataset"] = args.dataset
    final_results["num_train_envs"] = args.num_train_envs
    final_results["seed"] = args.seed
    final_results["args_id"] = args_id
    final_results["args"] = vars(args_no_seed)

    results_path = os.path.join(results_dir, f"{md5_fname}.jsonl")
    with open(results_path, 'a') as f:
        f.write(json.dumps(final_results, sort_keys=True, default=str) + "\n")
    print(f"\nResults saved to {results_path}")

    if use_wandb:
        wandb.log({f"final/{k}": v for k, v in final_results.items()
                   if isinstance(v, (int, float))})
        wandb.finish()
