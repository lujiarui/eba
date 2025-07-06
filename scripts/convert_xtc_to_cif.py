import argparse
from biotite.structure.io import load_structure, save_structure

import mdtraj, os, tempfile
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--atlas_dir', type=str) # raw trajectory 
parser.add_argument('--split', type=str, default=None, help='Path to the split file containing names of proteins')
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)
mmcif_outdir = os.path.join(args.outdir, 'mmcif')
os.makedirs(mmcif_outdir, exist_ok=True)

if args.split is None:
    targets = Path(args.atlas_dir).glob('*')
    jobs = [f.name for f in targets if f.is_dir()]
else:
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        #if os.path.exists(f'{args.outdir}/{name}.npz'): continue
        jobs.append(name)

print(f"Found {len(jobs)} targets to process.")


def pdb_to_cif(pdb_path, cif_path):
    pdb_structure = load_structure(pdb_path)
    save_structure(cif_path, pdb_structure)

def load_trajectory(name):
    traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
    # never load pdb as part of structure 
    # may contain invalid atom name
    ref = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.pdb')
    traj = ref + traj
    return traj     # mdtraj object

def convert_traj_to_cif(
    pc_name: str,   # xxxx_Y
):
    # load trajectory
    traj = load_trajectory(pc_name)
    save_idx = 0
    for i in trange(0, len(traj), 100):    # 1ns interval
        if i == 0:
            continue
        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp:
            traj[i].save_pdb(temp.name)
            # convert to cif 
            cif_path = os.path.join(mmcif_outdir, f"{pc_name}_{save_idx}.cif")
            save_idx += 1
            pdb_to_cif(temp.name, cif_path) # cif is saved
    return True


def main():
    if args.num_workers > 1:
        _ = [
            r for r in tqdm(
                Parallel(n_jobs=args.num_workers, return_as="generator_unordered")(
                    delayed(convert_traj_to_cif)(name)
            for name in jobs
            ),
            total=len(jobs),
        )
        ]
    else:
        for name in tqdm(jobs):
            convert_traj_to_cif(name)    
    return


main()