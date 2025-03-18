# Use this script on /Scripts directory

import os
import argparse
import asyncio
from asyncio.subprocess import create_subprocess_exec
from pathlib import Path
from itertools import cycle


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--version', type=str, default='v0.10')
    parser.add_argument('--scenario', type=str, default='inner', choices=['inner', 'e2e', 'old'])
    parser.add_argument('--component-order', type=str, default='TSRB_SRTB')
    parser.add_argument('--dataset', type=str, default='SFU', choices=['SFU', 'TVD'])
    parser.add_argument('--ctc-path', type=Path, default=Path.home() / 'Workspace/Project_VCM/vcm-ctc', help='Path to vcm-ctc')
    parser.add_argument('--input-path', type=Path, default=Path.home() / 'Workspace/Project_VCM/VCM-RS/Data')
    parser.add_argument('--output-path', type=Path, default=Path.home() / 'Workspace/Project_VCM/VCM-RS/Scripts/output')
    parser.add_argument('--gpu-ids', type=str, default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--seq-ids', type=str, default='1', help='Comma-separated sequence IDs')
    parser.add_argument('--configs', type=str, default='RA', help='Comma-separated configurations')
    parser.add_argument('--selection-algorithm', type=str, default='algo1')
    parser.add_argument('--experiment-suffix', type=str, default='')
    parser.add_argument('--skip-rs', action='store_true', help='Skip running VCM-RS')
    parser.add_argument('--num-worker-per-gpu', type=int, default=10)
    parser.add_argument('--pre-domain', type=str, default='RGB', choices=['RGB', 'YUV'])
    parser.add_argument('--post-domain', type=str, default='RGB', choices=['RGB', 'YUV'])
    parser.add_argument('--pre-model', type=str, default='FilterV8')
    parser.add_argument('--post-model', type=str, default='FilterV8')
    parser.add_argument('--bdr-width', type=int, default=1920)
    parser.add_argument('--bdr-height', type=int, default=1080)
    return parser.parse_args()


async def run(cmd, cwd=None, delay=0):
    await asyncio.sleep(delay)
    env = os.environ.copy()
    proc = await create_subprocess_exec(*cmd.split(), cwd=cwd, stdin=asyncio.subprocess.PIPE, env=env)
    await proc.wait()


async def main(args):
    result_dir = Path('./results')
    result_dir.mkdir(parents=True, exist_ok=True)

    experiment = f'{args.version}-{args.component_order}'
    if args.experiment_suffix:
        experiment += f'-{args.experiment_suffix}'

    result_excel_path = result_dir / 'excel' / f'{experiment}.xlsm'
    result_excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Run VCM-RS
    cmd = f'''
    python run.py
        --version {args.version} --scenario {args.scenario}
        --component-order {args.component_order} --dataset {args.dataset}
        --input-path {args.input_path} --output-path {args.output_path}
        --gpu-ids {args.gpu_ids} --seq-ids {args.seq_ids}
        --configs {args.configs} --selection-algorithm {args.selection_algorithm}
        --num-worker-per-gpu {args.num_worker_per_gpu}
        --pre-domain {args.pre_domain} --post-domain {args.post_domain} --pre-model {args.pre_model} --post-model {args.post_model}
        --bdr-width {args.bdr_width} --bdr-height {args.bdr_height}
        {"--experiment-suffix " + args.experiment_suffix if args.experiment_suffix else ""}'''
    if not args.skip_rs:
        await run(cmd)

    # Run vcm-ctc
    ctc_script_dir = args.ctc_path.resolve() / 'eval_scripts'
    gpu_id_iter = cycle(args.gpu_ids.split(','))
    tasks = []
    if args.dataset == 'SFU':
        tasks.append(asyncio.create_task(run(f'bash eval_sfu.sh SFU_AI_{args.scenario} {next(gpu_id_iter)} {args.output_path}/{experiment} {result_excel_path.resolve()}', cwd=ctc_script_dir, delay=0)))
        tasks.append(asyncio.create_task(run(f'bash eval_sfu.sh SFU_RA_{args.scenario} {next(gpu_id_iter)} {args.output_path}/{experiment} {result_excel_path.resolve()}', cwd=ctc_script_dir, delay=0)))
        tasks.append(asyncio.create_task(run(f'bash eval_sfu.sh SFU_LD_{args.scenario} {next(gpu_id_iter)} {args.output_path}/{experiment} {result_excel_path.resolve()}', cwd=ctc_script_dir, delay=0)))
    else:
        tasks.append(asyncio.create_task(run(f'bash eval_tvd_video.sh TVD_tracking_AI_{args.scenario} {next(gpu_id_iter)} {args.output_path}/{experiment} {result_excel_path.resolve()}', cwd=ctc_script_dir, delay=0)))
        tasks.append(asyncio.create_task(run(f'bash eval_tvd_video.sh TVD_tracking_RA_{args.scenario} {next(gpu_id_iter)} {args.output_path}/{experiment} {result_excel_path.resolve()}', cwd=ctc_script_dir, delay=0)))
        tasks.append(asyncio.create_task(run(f'bash eval_tvd_video.sh TVD_tracking_LD_{args.scenario} {next(gpu_id_iter)} {args.output_path}/{experiment} {result_excel_path.resolve()}', cwd=ctc_script_dir, delay=0)))
    await asyncio.gather(*tasks)
    # await run(f'python add_anchor_to_test.py {result_excel_path.resolve()}')


if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))
