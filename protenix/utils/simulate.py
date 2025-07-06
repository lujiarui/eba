"""
MD simulations based on pdbfixer and OpenMM.
"""

import io
import os
from pathlib import Path
import sys
from time import strftime, time
import logging
from typing import Collection, Optional, Sequence, Union
import tempfile
import shutil 
from glob import glob
import argparse
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import openmm
from openmm import unit
from openmm import app as openmm_app

from pdbfixer import PDBFixer


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
    pdbx_file: Path,
    save_to: Optional[Path] = None,
    add_hydrogens: bool = True, 
    remove_heterogens: bool = True,
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
    return pdbx


class EnergyMinimizer:
    def __init__(
        self,  
        parsed_file: Union[openmm_app.PDBFile, openmm_app.PDBxFile],
        temperature: float = 300,
        friction: float = 1.0,
        timestep_in_ps: float = 0.0025, # 2.5 fs
        implicit_solvent: bool = False,
        platform_name = 'CPU', 
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
        platform = openmm.Platform.getPlatformByName(platform_name)
    
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
            ionicStrength = 0 * unit.molar
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
        ionicStrength = 0 * unit.molar
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

    def __call__(self, batch_positions, tolerance: float = 1.0, max_iter: int = 0):
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
