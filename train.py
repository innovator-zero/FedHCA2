import argparse
import copy
import datetime
import os
import shutil
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from aggregate import aggregate, update_hyperweight
from datasets.custom_dataset import get_dataloader, get_dataset
from losses import get_criterion
from models.build_models import build_model
from models.hyperweight import HyperAggWeight, HyperCrossAttention
from train_utils import eval_metric, local_train
from utils import (
    RunningMeter,
    create_results_dir,
    get_loss_metric,
    get_mt_config,
    get_st_config,
    move_ckpt,
    set_seed,
)


def main(args, all_clients, hyperweight=None, local_rank=0):
    N = len(all_clients)
    # Setup loss meters
    train_loss = {}
    val_loss = {}
    for idx in range(N):
        train_loss[idx] = {}
        val_loss[idx] = {}
        for task in all_clients[idx]['tasks']:
            train_loss[idx][task] = RunningMeter()
            val_loss[idx][task] = RunningMeter()

    # Save last_ckpt
    last_ckpt = []
    for idx in range(N):
        last_ckpt.append(copy.deepcopy(all_clients[idx]['model'].module.state_dict()))
    if args.save_vram:
        last_ckpt = move_ckpt(last_ckpt, 'cpu')
    save_ckpt = copy.deepcopy(last_ckpt)

    # Create hyperweight log
    if local_rank == 0:
        if args.encoder_agg == "conflict_averse":
            if args.save_vram:
                enc_hw = hyperweight['enc']
            else:
                enc_hw = hyperweight['enc'].module
            alpha = enc_hw.alpha.detach().cpu().numpy().tolist()
            # log into file
            with open(os.path.join(args.exp_dir, 'enc_alpha.txt'), 'w') as f:
                f.write(str(alpha) + '\n')

        if args.decoder_agg == "cross_attention":
            if args.save_vram:
                dec_hw = hyperweight['dec']
            else:
                dec_hw = hyperweight['dec'].module
            beta = dec_hw.beta
            beta_list = [beta[key].detach().cpu().numpy().tolist() for key in dec_hw.beta_names]
            # log into file
            with open(os.path.join(args.exp_dir, 'dec_beta.txt'), 'w') as f:
                f.write(str(dec_hw.beta_names) + '\n')
                f.write(str(beta_list) + '\n')

    for cr in range(args.max_rounds):
        start_time = time.time()
        logs = {}
        for idx in range(N):
            # Train clients' local models for local epochs
            local_train(idx=idx,
                        cr=cr,
                        train_loss=train_loss[idx],
                        local_rank=local_rank,
                        fp16=args.fp16,
                        **all_clients[idx])

            train_stats = get_loss_metric(train_loss[idx], all_clients[idx]['tasks'], 'train', idx)
            logs.update(train_stats)

        # Update save_ckpt
        for idx in range(N):
            save_ckpt[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
        if args.save_vram:
            save_ckpt = move_ckpt(save_ckpt, 'cpu')

        # Update hyperweight
        if cr > 0:
            update_hyperweight(all_clients, hyperweight, save_ckpt, last_ckpt)
            # Update hyperweight log
            if local_rank == 0:
                if args.encoder_agg == "conflict_averse":
                    if args.save_vram:
                        enc_hw = hyperweight['enc']
                    else:
                        enc_hw = hyperweight['enc'].module
                    alpha = enc_hw.alpha.detach().cpu().numpy().tolist()
                    with open(os.path.join(args.exp_dir, 'enc_alpha.txt'), 'a') as f:
                        f.write(str(alpha) + '\n')

                if args.decoder_agg == "cross_attention":
                    if args.save_vram:
                        dec_hw = hyperweight['dec']
                    else:
                        dec_hw = hyperweight['dec'].module
                    beta = dec_hw.beta
                    beta_list = [beta[key].detach().cpu().numpy().tolist() for key in dec_hw.beta_names]
                    with open(os.path.join(args.exp_dir, 'dec_beta.txt'), 'a') as f:
                        f.write(str(beta_list) + '\n')

        # Aggregate clients' models
        aggregate(
            all_clients,
            save_ckpt,
            last_ckpt,
            hyperweight,
            args.encoder_agg,
            args.decoder_agg,
            args.ca_c,
        )

        # Update last_ckpt
        for idx in range(N):
            last_ckpt[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
        if args.save_vram:
            last_ckpt = move_ckpt(last_ckpt, 'cpu')

        if local_rank == 0:
            print("CR %d finishs, Time: %.1fs." % (cr, time.time() - start_time))

            if (cr + 1) == args.max_rounds or (cr + 1) % args.eval_freq == 0:
                print('Validation at CR %d.' % cr)
                # Evaluation on metrics
                val_logs = {}
                for idx in range(N):
                    res = eval_metric(idx=idx, **all_clients[idx])
                    val_logs.update(res)
                print(val_logs)
                if args.wandb_name is not None:
                    wandb.log({**logs, **val_logs})

                # Save checkpoint
                save_ckpt_temp = {}
                for idx in range(N):
                    save_ckpt_temp[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
                torch.save(save_ckpt_temp, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
                print('Checkpoint saved.')
                del save_ckpt_temp
            else:
                if args.wandb_name is not None:
                    wandb.log(logs)

    if local_rank == 0:
        print('Training finished.')


def get_clients(args, model_config, client_configs, local_rank):
    """
    Get clients from configs
    """

    all_clients = []
    n_decoders = 0

    for dataname in client_configs:
        client_config = client_configs[dataname]
        net_task_dataidx_map, n_clients = (
            client_config['net_task_dataidx_map'],
            client_config['n_clients'],
        )

        for idx in range(n_clients):
            task_list = net_task_dataidx_map[idx]['task_list']
            dataidxs = net_task_dataidx_map[idx]['dataidx']

            # Setup dataset and dataloader
            train_ds_local = get_dataset(
                dataname=dataname,
                train=True,
                tasks=task_list,
                transform=client_config['train_transforms'],
                dataidxs=dataidxs,
                local_rank=local_rank,
            )
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds_local, drop_last=True)
            train_dl_local = get_dataloader(
                train=True,
                configs=client_config,
                dataset=train_ds_local,
                sampler=train_sampler,
            )

            val_ds_local = get_dataset(
                dataname=dataname,
                train=False,
                tasks=task_list,
                transform=client_config['val_transforms'],
                local_rank=local_rank,
            )
            val_dl_local = get_dataloader(train=False, configs=client_config, dataset=val_ds_local)

            # Setup model
            model = build_model(task_list, dataname, **model_config).cuda()
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank])

            # Setup optimizer and scheduler
            if client_config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=float(client_config['lr']),
                    momentum=0.9,
                    weight_decay=float(client_config['weight_decay']),
                )
            elif client_config['optimizer'] == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(client_config['lr']),
                    weight_decay=float(client_config['weight_decay']),
                )
            else:
                raise NotImplementedError("Invalid optimizer %s!" % client_config['optimizer'])

            max_epochs = int(args.max_rounds) * int(client_config['local_epochs'])
            warmup_epochs = int(client_config['warmup_epochs'])
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=max_epochs - warmup_epochs,
                lr_min=1.25e-6,
                warmup_t=warmup_epochs,
                warmup_lr_init=1.25e-7,
                warmup_prefix=True,
            )
            client = {}
            client['tasks'] = task_list
            client['dataname'] = dataname
            client['train_dl'] = train_dl_local
            client['val_dl'] = val_dl_local
            client['local_epochs'] = client_config['local_epochs']
            client['model'] = model
            client['optimizer'] = optimizer
            client['scheduler'] = scheduler
            # Setup loss function
            client['criterion'] = get_criterion(dataname, task_list).cuda()
            # Setup scaler for amp
            client['scaler'] = torch.cuda.amp.GradScaler(enabled=args.fp16)

            all_clients.append(client)
            n_decoders += len(task_list)

    return all_clients, n_decoders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Config file path")
    parser.add_argument('--exp', type=str, required=True, help="Experiment name")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, help="Wandb project name")
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
    parser.add_argument('--save_vram', action='store_true', help='Whether to save vram')  # move aggregation to cpu

    parser.add_argument('--max_rounds', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=4)

    parser.add_argument('--encoder_agg', default='conflict_averse', help="none,fedavg")
    parser.add_argument('--ca_c', type=float, default=0.4)
    parser.add_argument('--enc_alpha_init', type=float, default=0.1)
    parser.add_argument('--decoder_agg', default='cross_attention', help="none,fedavg")
    parser.add_argument('--dec_beta_init', type=float, default=0.1)

    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        exp_config = yaml.safe_load(stream)

    # Join args and configs
    exp_config = {**exp_config, **vars(args)}

    # Set seed and ddp
    set_seed(args.seed)
    dist.init_process_group('nccl', timeout=datetime.timedelta(0, 3600 * 2))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    # Setup logger and output folders
    if local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        args.exp_dir, args.checkpoint_dir = create_results_dir(args.results_dir, args.exp)
        shutil.copy(args.config_path, os.path.join(args.exp_dir, 'config.yml'))
        if args.wandb_name is not None:
            import wandb

            wandb.init(project=args.wandb_name, id=args.exp, name=args.exp, config=exp_config)
    dist.barrier()

    # Get single-task and multi-task config
    client_configs = {}
    if 'ST_Datasets' in exp_config:
        client_configs.update(get_st_config(exp_config['ST_Datasets'], local_rank))

    if 'MT_Datasets' in exp_config:
        client_configs.update(get_mt_config(exp_config['MT_Datasets'], local_rank))

    # Get all clients
    all_clients, n_decoders = get_clients(args, exp_config['Model'], client_configs, local_rank)

    # Get hyperweight
    hyperweight = {}
    if args.encoder_agg == "conflict_averse":
        hypernet = HyperAggWeight(K=len(all_clients), init_alpha=args.enc_alpha_init)
        if args.save_vram:
            hyperweight['enc'] = hypernet  # on cpu
        else:
            hyperweight['enc'] = DDP(hypernet.cuda(), device_ids=[local_rank])  # on gpu
        hyperweight['enc_optimizer'] = torch.optim.SGD(hypernet.parameters(), **exp_config['Hyperweight'])

    if args.decoder_agg == "cross_attention":
        dummy_decoder = all_clients[0]['model'].module.decoders
        hypernet = HyperCrossAttention(model=dummy_decoder, K=n_decoders, init_beta=args.dec_beta_init)
        if args.save_vram:
            hyperweight['dec'] = hypernet  # on cpu
        else:
            hyperweight['dec'] = DDP(hypernet.cuda(), device_ids=[local_rank])  # on gpu
        hyperweight['dec_optimizer'] = torch.optim.SGD(hypernet.parameters(), **exp_config['Hyperweight'])

    main(
        args=args,
        all_clients=all_clients,
        hyperweight=hyperweight,
        local_rank=local_rank,
    )
    dist.destroy_process_group()
