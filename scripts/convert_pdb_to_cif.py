import argparse
from biotite.structure.io import load_structure, save_structure

import mdtraj, os, tempfile
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--pdb_dir', type=str) # raw trajectory 
parser.add_argument('--outdir', type=str, default='./example/mmcif')
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()


os.makedirs(outdir, exist_ok=True)
targets = Path(args.pdb_dir).glob('*')
jobs = [
    str(target) for target in targets if str(target).endswith('.pdb')
]
print(f"Found {len(jobs)} PDB targets to process.")


def pdb_to_cif(pdb_path, cif_path=None):
    name = os.path.basename(pdb_path).replace('.pdb', '')
    pdb_structure = load_structure(pdb_path)
    if cif_path is None:
        cif_path = os.path.join(mmcif_outdir, f"{name}.cif")
    save_structure(cif_path, pdb_structure)

def main():
    if args.num_workers > 1:
        _ = [
            r for r in tqdm(
                Parallel(n_jobs=args.num_workers, return_as="generator_unordered")(
                    delayed(pdb_to_cif)(name)
            for name in jobs
            ),
            total=len(jobs),
        )
        ]
    else:
        for name in tqdm(jobs):
            pdb_to_cif(name)    
    return

main()