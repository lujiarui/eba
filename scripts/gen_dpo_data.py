"""Stand-alone script for MD simulations based on pdbfixer and OpenMM."""

import io, os, sys
from pathlib import Path
from time import strftime, time
import logging
from typing import Collection, Optional, Sequence, Union
import tempfile
import shutil 
from glob import glob
import argparse
from functools import partial
import multiprocessing as mp

from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import pandas as pd

import openmm
from openmm import unit
from openmm import app as openmm_app
from pdbfixer import PDBFixer

# num_workers = mp.cpu_count()
logging.basicConfig(level=logging.INFO)


#######################################
# Physical units 
GLOBAL_DEVICE_ID = None
ENERGY = unit.kilojoule_per_mole # unit.kilocalories_per_mole
LENGTH = unit.angstroms # unit.nanometers
TEMPERATURE = unit.kelvin
TIME = unit.picoseconds
PRESSURE = unit.atmospheres
BOLTZMANN_CONST = 0.001985875 # unit kcal/mol K
print(
    "Physical units: ",
    f"Energy: {ENERGY}, Length: {LENGTH}, Temperature: {TEMPERATURE}, Time: {TIME}, Pressure: {PRESSURE}",
)
#######################################

def parse_clean_pdbx(
    pdbx_file: str,
    save_to: Optional[Path] = None,
    add_hydrogens: bool = True, 
    remove_heterogens: bool = True,
    return_dict: bool = False,
):
    """
    Apply pdbfixer to the contents of a PDB file; return a PDB string result.
    
    *** This function will only process the first model in the PDB file ***
    
    Example inspired by https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html.
    
    1) Replaces nonstandard residues.
    2) Removes heterogens (non protein residues) including water.
    3) Adds missing residues and missing atoms within existing residues.
    4) Adds hydrogens assuming pH=7.0.
    5) KeepIds is currently true, so the fixer must keep the existing chain and
        residue identifiers. This will fail for some files in wider PDB that have
        invalid IDs.
    By default, the input pdbfile contains single-chain structure CA array.
    """
    fixer = PDBFixer(filename=pdbx_file)
    # standardize the residue name
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()   # not do this
    # add side chains 
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
     # add hydrogens
    if add_hydrogens:
        fixer.addMissingHydrogens(7.0)  # necessary for minimization
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=False)   # remove heterogens including water
    # save to pdb string
    out_handle = io.StringIO()
    openmm_app.PDBxFile.writeFile(
        fixer.topology, 
        fixer.positions, 
        out_handle,
        keepIds=True,
    )
    pdbx_string = out_handle.getvalue()    # pdb string
    
    # Configure output directory.
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, 'w') as f: f.write(pdbx_string)
    # return pdbx_string
    pdbx_f = io.StringIO(pdbx_string)
    pdbx = openmm_app.PDBxFile(pdbx_f)
    if return_dict:
        return {pdbx_file: pdbx}
    return pdbx


class EnergyMinimizer:
    def __init__(
        self,  
        parsed_file: Union[openmm_app.PDBFile, openmm_app.PDBxFile],
        temperature: float = 300,
        friction: float = 1.0,
        timestep_in_ps: float = 0.0025, # 2.5 fs
        implicit_solvent: bool = False,
        platform_name = 'CUDA', 
        platform_properties = {},
        
        # restrain_coords: bool = False,
        # stiffness: float = 10.0,
        # restraint_set: str = "non_hydrogen",
        # exclude_residues: Optional[Collection[int]] = None,
    ):
        """Minimize energy via openmm.
            The snippets are inspired from http://docs.openmm.org/latest/userguide/application/02_running_sims.html.
        
        #  Default Langevin dynamics in OpenMM:
            #   the simulation temperature (298 K),
            #   the friction coefficient (1 ps-1),
            #   and the step size (4 fs).
        
        Args:
            stiffness: kcal/mol A**2, the restraint stiffness. 
                The default value is the AlphaFold default.
            friction: ps^-1, the friction coefficient for Langevin dynamics. 
                Unit of reciprocal time.
            temperature: kelvin, the temperature for simulation.
            timestep_in_ps: ps, the timestep_in_ps for Langevin dynamics. 
            
            restrain_coords: bool, whether to restrain the coordinates. 
                Set to True if you want to relax. (default: False)
        """
        self.implicit_solvent = implicit_solvent
        platform = openmm.Platform.getPlatform(platform_name)
    
        # assign physical units
        # exclude_residues = exclude_residues or []
        # stiffness = stiffness * ENERGY / (LENGTH**2)
        temperature = temperature * TEMPERATURE
        friction = friction / TIME
        timestep_in_ps = timestep_in_ps * TIME
        
        # create system and force field
        # forcefield_name = 'amber14-all.xml'
        # forcefield_name = 'amber14/protein.ff14SB.xml'
        forcefield_name = 'charmm36.xml'
        self.input_topology = parsed_file.topology
        
        if implicit_solvent:
            solvent_name = "implicit/gbn2.xml"
            forcefield = openmm_app.ForceField(forcefield_name, solvent_name) 
            self.forcefield = forcefield
            topology, positions = parsed_file.topology, parsed_file.positions
            system = forcefield.createSystem(topology, nonbondedMethod=openmm_app.NoCutoff,
                                            nonbondedCutoff=1*unit.nanometer, constraints=openmm_app.HBonds,
                                            soluteDielectric=1.0, solventDielectric=78.5)    
        else:
            # solvent_name = 'amber14/tip3p.xml'
            solvent_name = 'charmm36/water.xml'
            forcefield = openmm_app.ForceField(forcefield_name, solvent_name)
            self.forcefield = forcefield
            # add hydrogen
            modeller = openmm.app.Modeller(parsed_file.topology, parsed_file.positions)
            modeller.addHydrogens(forcefield, pH=7.0)
            topology, positions = modeller.getTopology(), modeller.getPositions()
            # add solvent (see http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html?highlight=padding)
            box_padding = 1.0 * unit.nanometers
            ionicStrength = 0.15 * unit.molar
            positiveIon = 'Na+' 
            negativeIon = 'Cl-'
            # modeller = openmm_app.Modeller(topology, positions)
            modeller.addSolvent(
                forcefield, 
                model='tip3p',
                # boxSize=openmm.Vec3(5.0, 5.0, 5.0) * unit.nanometers,
                padding=box_padding,
                ionicStrength=ionicStrength,
                positiveIon=positiveIon,
                negativeIon=negativeIon,
            )
            topology, positions = modeller.getTopology(), modeller.getPositions()
            system = forcefield.createSystem(
                topology, nonbondedMethod=openmm_app.PME, constraints=None, rigidWater=None
            )

        # add restraints if necessary
        # if restrain_coords and stiffness > 0 * ENERGY / (LENGTH**2):
            # _add_restraints(system, parsed_file, stiffness, restraint_set, exclude_residues)
        
        # see http://docs.openmm.org/latest/userguide/theory/04_integrators.html#integrators-theory for choice of integrators
        integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep_in_ps)
        if GLOBAL_DEVICE_ID is not None:
            platform.setPropertyDefaultValue(property='DeviceIndex', value=str(GLOBAL_DEVICE_ID))
        
        self.simulation = openmm_app.Simulation(topology, system, integrator, platform) 
        self.topology = topology
        self._pos_in_vacuum = parsed_file.positions
        self._positions = positions
        # simulation.context.setPositions(positions)  # assign positions, different between implicit and explicit solvent
        print(f"System has {system.getNumParticles()} particles in total | Platform: {platform}")
        # self._context = Context(system, integrator, platform, platform_properties)

    def _add_environment(self, positions):
        # add hydrogen
        modeller = openmm.app.Modeller(self.input_topology, positions)
        
        pos = modeller.getPositions()
        if self.implicit_solvent:
            return pos, pos
        
        modeller.addHydrogens(self.forcefield, pH=7.0)
        # add solvent (see http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html?highlight=padding)
        box_padding = 1.0 * unit.nanometers
        ionicStrength = 0.15 * unit.molar
        positiveIon = 'Na+' 
        negativeIon = 'Cl-'
        modeller.addSolvent(
            self.forcefield, 
            model='tip3p',
            # boxSize=openmm.Vec3(5.0, 5.0, 5.0) * unit.nanometers,
            padding=box_padding,
            ionicStrength=ionicStrength,
            positiveIon=positiveIon,
            negativeIon=negativeIon,
        )
        topology, positions = modeller.getTopology(), modeller.getPositions()
        return pos, positions

    def __call__(self, tolerance: float = 1.0, max_iter: int = 0):
        """Compute energies and/or forces.
        
        Default:
            tolerance: 1.0 kcal/mol/A
        """
        tolerance = tolerance * ENERGY / LENGTH
        # initialize state
        positions = self._positions
        self.simulation.context.setPositions(positions)
        state = self.simulation.context.getState(
            getEnergy=True,
            # getPositions=True
        )
        # pos_init = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        init_energy = state.getPotentialEnergy().value_in_unit(ENERGY)
        print(f"Initial energy: {init_energy}")
        self.simulation.minimizeEnergy(
            tolerance=tolerance, maxIterations=max_iter
        )

        # compute energy and forces
        state = self.simulation.context.getState(
            getEnergy=True,
            getForces=True,
            getPositions=True
        )
        # ready to output
        energy = state.getPotentialEnergy().value_in_unit(ENERGY)
        force = state.getForces(asNumpy=True).value_in_unit(ENERGY / LENGTH)
        
        pos_init = unit.Quantity(np.array(self._pos_in_vacuum.value_in_unit(LENGTH)), LENGTH)
        pos_final = unit.Quantity(np.array(state.getPositions().value_in_unit(LENGTH)), LENGTH)
        if not self.implicit_solvent:
            modeller = openmm.app.Modeller(self.topology, pos_final)
            # modeller.deleteWater()
            # print(f"Residues names: {set(res.name for res in self.topology.residues())}")
            modeller.delete(res for res in self.topology.residues() if res.name == "HOH" or res.name == "NA" or res.name == "CL")
            pos_final = unit.Quantity(np.array(modeller.getPositions().value_in_unit(LENGTH)), LENGTH)
            pos_init = unit.Quantity(np.array(pos_init.value_in_unit(LENGTH)), LENGTH)
        
        rmsd = np.sqrt(np.mean(np.sum(np.square(pos_final - pos_init), axis=-1)))
        # pos_final = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        # rmsd = np.sqrt(np.mean(np.sum(np.square(pos_final - pos_init), axis=-1)))
        return {
            "initial_energy": init_energy,
            "energy": energy,
            "force": force,
            "position": pos_final,
            "rmsd": rmsd,
        }

    def batch_call(self, batch_positions, tolerance: float = 1.0, max_iter: int = 0):
        """Compute energies and/or forces.
        
        Default:
            tolerance: 1.0 kcal/mol/A
        """
        tolerance = tolerance * ENERGY / LENGTH
        batch_init_energies = []
        batch_energies = []
        batch_forces = []
        batch_new_positions = []
        batch_rmsd = [] 
        # for loop over batch_positions
        for pos in batch_positions:
            pos, pos_in_env = self._add_environment(pos)
            try:
                # initialize state
                self.simulation.context.setPositions(pos_in_env)
                state = self.simulation.context.getState(
                    getEnergy=True,
                    # getPositions=True
                )
                # pos_init = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
                batch_init_energies.append(state.getPotentialEnergy().value_in_unit(ENERGY))

                self.simulation.minimizeEnergy(
                    tolerance=tolerance, maxIterations=max_iter
                )

                # compute energy and forces
                state = self.simulation.context.getState(
                    getEnergy=True,
                    getForces=True,
                    getPositions=True
                )
                # ready to output
                energy = state.getPotentialEnergy().value_in_unit(ENERGY)
                print(f"Energy: {energy}")
                force = state.getForces(asNumpy=True).value_in_unit(ENERGY / LENGTH)
                print(f"Force: {force}")
                batch_energies.append(energy)
                batch_forces.append(force)

                modeller = openmm.app.Modeller(self.topology, state.getPositions())
                # modeller.deleteWater()
                # print(f"Residues names: {set(res.name for res in self.topology.residues())}")
                modeller.delete(res for res in self.topology.residues() if res.name == "HOH" or res.name == "NA" or res.name == "CL")
                new_pos = unit.Quantity(np.array(modeller.getPositions().value_in_unit(LENGTH)), LENGTH)
                init_pos = unit.Quantity(np.array(pos.value_in_unit(LENGTH)), LENGTH)
                batch_new_positions.append(new_pos) 
                # print(f"New pos shape: {new_pos.shape} | Pos shape: {init_pos.shape}")
                rmsd = np.sqrt(np.mean(np.sum(np.square(new_pos - init_pos), axis=-1)))
                batch_rmsd.append(rmsd)
                
            except Exception as e:
                print(f"Suppressed exception: {e}")
                raise e
        
        batch_init_energies = np.array(batch_init_energies)
        batch_energies = np.array(batch_energies)
        batch_rmsd = np.array(batch_rmsd)

        return {
            "initial_energies": batch_init_energies,
            "energies": batch_energies,
            "forces": batch_forces,
            "positions": batch_new_positions,
            "rmsd": batch_rmsd,
        }

# CPU job for cleaning
def batch_parse_clean_pdbx(
    pdbx_files: Sequence[str],
    save_to: Optional[Path] = None,
    num_workers: int = 32,
):
    print(f"To clean and convert {len(pdbx_files)} cif files to CIF format.")

    if num_workers > 1:
        name_to_pdbx_str = [
            r for r in tqdm(
                Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                    delayed(parse_clean_pdbx)(pdbx_file, return_dict=True)
            for pdbx_file in pdbx_files
            ),
            total=len(pdbx_files),
        )
        ]
        pdbx_str = {k: v for d in name_to_pdbx_str for k, v in d.items()}

    else:
        pdbx_str = {}
        for pdbx_file in tqdm(pdbx_files):
            pdbx_str[pdbx_file] = parse_clean_pdbx(pdbx_file)
    return pdbx_str


def process_one(pdbxfile, name, output_dir, implicit_solvent=False):
    print("Use implicit_solvent?", implicit_solvent)
    start_t = time()
    sim = EnergyMinimizer(parsed_file=pdbxfile, implicit_solvent=implicit_solvent) 
    print(f"Created system using example pdbx file. Time elapsed: {time() - start_t:.2f}s")
    start_t = time()
    res = sim(tolerance=1)  # 10 kcal/mol/A = 1 kcal/mol/nm
    print(f"Minimization done. Time elapsed: {time() - start_t:.2f}s")
    print(
        f"Initial energy: {res['initial_energy']:.2f} kj/mol | Final energy: {res['energy']:.2f} kj/mol | RMSD: {res['rmsd']:.2f} A"
    )
    namebase= os.path.basename(name)
    with open(output_dir / f"{namebase}_min.cif", 'w') as fo:
        openmm_app.PDBxFile.writeFile(sim.input_topology, res['position'], fo)
    return {
        "initial_e": res['initial_energy'],
        "final_e": res['energy'],
        "rmsd": res['rmsd'],
        "name": name,
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Run energy minimization on a set of PDB files using OpenMM."
    )
    args.add_argument(
        '-i', "--input_dir", type=Path, default=Path("~/scratch/data/atlas/data_atlas/mmcif/").expanduser(),
        help="Directory containing the input PDB files to be processed."
    )
    args.add_argument(
        '-c', "--num_workers", type=int, default=mp.cpu_count(),
        help="Number of worker processes to use for parallel processing. Default is the number of CPU cores."
    )
    
    args = args.parse_args()
    input_dir = args.input_dir
    output_dir = input_dir.parent / "minimized"
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [f.name for f in input_dir.glob("*")]
    print(f"Found {len(names)} targets.")
    cif_files = [str(f) for f in input_dir.glob("**/*.cif")]
    
    ### if sub directories are used
    # for name in names:
        # name_sample_dir = input_dir / f"{name}"
        # cif_files.extend([str(f) for f in name_sample_dir.glob("**/*.cif")])

    ### for debugging
    # num_workers = 4 
    # cif_files = cif_files[:4]

    print(f"Cleaning {len(cif_files)} files with {args.num_workers} workers.")
    name_to_pdbx = batch_parse_clean_pdbx(cif_files, output_dir, num_workers=args.num_workers)
    print(f"cleaned {len(name_to_pdbx)} pdbx files")
    example_pdbx = list(name_to_pdbx.values())[0]
    
    # ========================
    with open(output_dir / "openmm_labels.csv", 'a') as f:
        f.write("name,initial_e,final_e,rmsd\n")
        
    res_dict = {}
    for name, pdbxfile in name_to_pdbx.items():
        res = process_one(pdbxfile, name, output_dir, implicit_solvent=True)
        if res_dict == {}:
            res_dict = {k: [v] for k, v in res.items()}
        else:
            for k, v in res.items():
                res_dict[k].append(v)
        with open(output_dir / "openmm_labels.csv", 'a') as f:
            f.write(f"{name},{res['initial_e']},{res['final_e']},{res['rmsd']}\n")
    
    # # save to output, index=name
    df = pd.DataFrame(res_dict, index=[os.path.basename(name) for name in name_to_pdbx.keys()])
    df.to_csv(output_dir / "minimization_results.csv")
