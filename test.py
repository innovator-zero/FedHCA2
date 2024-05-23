import argparse
import os

import torch
import yaml
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from evaluation.evaluate_utils import PerformanceMeter, predict
from models.build_models import build_model
from utils import create_pred_dir, get_mt_config, get_output, get_st_config, to_cuda


def eval_metric(idx, dataname, tasks, test_dl, model, evaluate, save, pred_dir, **args):
    '''
    Evalution of the model
    '''

    if evaluate:
        performance_meter = PerformanceMeter(dataname, tasks)

    if save:
        # save predictions of all tasks
        tasks_to_save = tasks
    else:
        # save only predictions of edge
        tasks_to_save = ['edge'] if 'edge' in tasks else []

    assert evaluate or len(tasks_to_save) > 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating Net %d Task: %s" % (idx, ",".join(tasks))):
            batch = to_cuda(batch)
            images = batch['image']
            outputs = model(images)

            if evaluate:
                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

            for task in tasks_to_save:
                predict(dataname, batch['meta'], outputs, task, pred_dir, idx)

    if evaluate:
        # get evaluation results
        eval_results = performance_meter.get_score()

        results_dict = {}
        for t in tasks:
            for key in eval_results[t]:
                results_dict[str(idx) + '_' + t + '_' + key] = eval_results[t][key]

        return results_dict


def test(args, all_clients):
    '''
    Test all clients with test data
    '''
    test_results = {}
    for idx in range(len(all_clients)):
        res = eval_metric(idx=idx, evaluate=args.evaluate, save=args.save, pred_dir=args.pred_dir, **all_clients[idx])
        if args.evaluate:
            test_results.update({key: "%.5f" % res[key] for key in res})

    # log results
    if args.evaluate:
        print(test_results)
        results_file = os.path.join(args.results_dir, args.exp, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write(str(test_results))


def get_clients(client_configs, model_config):
    """
    Get clients from configs
    """

    all_clients = []

    for dataname in client_configs:
        client_config = client_configs[dataname]
        net_task_dataidx_map, n_clients = (
            client_config['net_task_dataidx_map'],
            client_config['n_clients'],
        )

        for idx in range(n_clients):
            task_list = net_task_dataidx_map[idx]['task_list']

            test_ds_local = get_dataset(
                dataname=dataname,
                train=False,
                tasks=task_list,
                transform=client_config['val_transforms'],
            )
            test_dl_local = get_dataloader(train=False, configs=client_config, dataset=test_ds_local)

            model = build_model(task_list, dataname, **model_config).cuda()

            client = {}
            client['tasks'] = task_list
            client['dataname'] = dataname
            client['test_dl'] = test_dl_local
            client['model'] = model

            all_clients.append(client)

    return all_clients


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='experiment name')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('--evaluate', action='store_true', help='Whether to evaluate all clients')
    parser.add_argument('--save', action='store_true', help='Whether to save predictions')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')

    args = parser.parse_args()

    with open(os.path.join(args.results_dir, args.exp, 'config.yml'), 'r') as stream:
        exp_config = yaml.safe_load(stream)

    # Set gpu
    torch.cuda.set_device(args.gpu_id)

    # Get single-task and multi-task config
    client_configs = {}
    if 'ST_Datasets' in exp_config:
        client_configs.update(get_st_config(exp_config['ST_Datasets']))

    if 'MT_Datasets' in exp_config:
        client_configs.update(get_mt_config(exp_config['MT_Datasets']))

    # Get all clients
    all_clients = get_clients(client_configs, exp_config['Model'])

    # Setup output folders
    args.checkpoint_dir, args.pred_dir = create_pred_dir(args.results_dir, args.exp, all_clients)

    # Load model from checkpoint
    checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_file):
        raise ValueError('Checkpoint %s not found!' % (checkpoint_file))

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    for idx in range(len(all_clients)):
        all_clients[idx]['model'].load_state_dict(checkpoint[idx])

    test(args, all_clients)
