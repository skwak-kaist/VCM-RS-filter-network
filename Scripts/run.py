# Use this script on /Scripts directory


import os
import asyncio
import argparse
from itertools import cycle
from pathlib import Path
from asyncio.subprocess import create_subprocess_exec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v0.10')
    parser.add_argument('--scenario', type=str, default='inner', choices=['inner', 'e2e', 'old'])
    parser.add_argument('--component-order', type=str, default='TSRB_SRTB')
    parser.add_argument('--dataset', type=str, default='SFU', choices=['SFU', 'TVD'])
    parser.add_argument('--input-path', type=Path, default=Path.home() / 'data/VCM_Data')
    parser.add_argument('--output-path', type=Path, default=Path.home() / 'data/VCM_Output/moon')
    parser.add_argument('--experiment-suffix', type=str, default='')
    parser.add_argument('--gpu-ids', type=str, default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--seq-ids', type=str, default='1', help='Comma-separated sequence IDs')
    parser.add_argument('--configs', type=str, default='RA', help='Comma-separated configurations')
    parser.add_argument('--selection-algorithm', type=str, default='algo1')
    parser.add_argument('--num-worker-per-gpu', type=int, default=10)
    parser.add_argument('--pre-domain', type=str, default='RGB', choices=['RGB', 'YUV'])
    parser.add_argument('--post-domain', type=str, default='RGB', choices=['RGB', 'YUV'])
    parser.add_argument('--pre-model', type=str, default='FilterV8')
    parser.add_argument('--post-model', type=str, default='FilterV8')
    parser.add_argument('--bdr-width', type=int, default=1920)
    parser.add_argument('--bdr-height', type=int, default=1080)
    return parser.parse_args()


async def main(args):
    num_workers = args.num_worker_per_gpu * len(args.gpu_ids.split(','))
    gpu_id_iter = cycle(map(int, args.gpu_ids.split(',')))
    dataset = args.dataset
    scenario = args.scenario
    component_order = args.component_order
    selection_algorithm = args.selection_algorithm
    pre_domain = args.pre_domain
    post_domain = args.post_domain
    pre_model = args.pre_model
    post_model = args.post_model
    bdr_width = args.bdr_width
    bdr_height = args.bdr_height

    print("################################################")
    print('component_order:', component_order)
    print('selection_algorithm:', selection_algorithm)
    print('pre_domain:', pre_domain)
    print('post_domain:', post_domain)
    print("################################################")


    experiment = f'{args.version}-{args.component_order}'
    if args.experiment_suffix:
        experiment += f'-{args.experiment_suffix}'

    output_path = args.output_path / experiment
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = Path('../Data')
    if not data_path.exists():
        data_path.symlink_to(args.input_path, target_is_directory=True)

    # Run processes parallelly
    semaphore = asyncio.Semaphore(num_workers)
    seq_ids = [int(id_) for id_ in args.seq_ids.split(',')]
    configs = args.configs.split(',')
    task_infos = get_remaining_task_infos(dataset, output_path, seq_ids, configs, scenario)
    print(f'total: {len(task_infos)}')
    await asyncio.sleep(5)
    tasks = []
    for cfg, task_id in task_infos:
        task = asyncio.create_task(encode(dataset, scenario, cfg, task_id, gpu_id_iter, output_path, component_order, selection_algorithm, pre_domain, post_domain, pre_model, post_model, bdr_width, bdr_height, semaphore))
        tasks.append(task)
    await asyncio.gather(*tasks)
    print('All sequences are processed')


async def encode(dataset, scenario, cfg, task_id, gpu_id_iter, output_path, component_order, selection_algorithm, pre_domain, post_domain, pre_model, post_model, bdr_width, bdr_height, semaphore):
    assert scenario in ['inner', 'e2e', 'old']
    assert cfg in ['AI', 'RA', 'LD']
    env = os.environ.copy()

    scenario = f'{cfg}_{scenario}'

    if dataset == 'SFU':
        cmd = f"python VTM_InnerCodec/encode_sfu.py {task_id} {scenario} {output_path} {component_order} {selection_algorithm} {pre_domain} {post_domain} {pre_model} {post_model} {bdr_width} {bdr_height}"
    elif dataset == 'TVD':
        cmd = f"python VTM_InnerCodec/encode_tvd_tracking.py {task_id} {scenario} {output_path} {component_order} {selection_algorithm} {pre_domain} {post_domain} {pre_model} {post_model} {bdr_width} {bdr_height}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    while True:
        gpu_id = next(gpu_id_iter)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        async with semaphore:
            proc = await create_subprocess_exec(
                *cmd.split(), env=env,
                stdin=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if proc.returncode == 0:
                break
        await asyncio.sleep(30)


def get_remaining_task_infos(dataset, output_path, seq_ids, configs, scenario):
    import sfu_config
    import tvd_tracking_config

    optional_seqs = ['Kimono', 'Cactus']
    
    qp_indices = [0, 1, 2, 3, 4, 5]
    if dataset == 'SFU':
        recon_names = [v[1] for v in sfu_config.seq_dict.values()]
    elif dataset == 'TVD':
        recon_names = [k for k in tvd_tracking_config.fr_dict.keys()]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    step = len(recon_names)

    infos = []
    for cfg in configs:
        prefix = dataset if dataset == 'SFU' else f'{dataset}_tracking'
        recon_path = output_path / f'{prefix}_{cfg}_{scenario}/recon'
        for start_id, name in enumerate(recon_names, start=1):
            if start_id not in seq_ids: continue
            for qpi in qp_indices:
                recon_file = recon_path / f'{name}_qp{qpi}.yuv' if dataset == 'SFU' else recon_path / f'qp{qpi}/{name}.yuv'
                if (not recon_file.exists()) and any(s not in name for s in optional_seqs):
                    print(f'Queue sequence: {cfg} {name} {qpi}')
                    id_ = start_id + qpi * step
                    infos.append((cfg, id_))
    infos.sort(key=lambda x: x[1])
    return infos


if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))