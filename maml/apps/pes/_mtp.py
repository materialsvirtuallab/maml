# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This module provides MTP interatomic potential class."""

import itertools
import json
import os
import re
import shutil
import subprocess
from collections import OrderedDict

import numpy as np
from monty.io import zopen
from monty.os.path import which
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.core import Structure, Lattice

from maml.utils import pool_from, convert_docs, check_structures_forces_stresses
from maml.apps.pes._base import Potential
from maml.apps.pes._lammps import EnergyForceStress

module_dir = os.path.dirname(__file__)
MTini_params = loadfn(os.path.join(module_dir, "params", "MTini.json"))


def feed(attribute, kwargs, dictionary, tab="\t"):
    r"""
    Args:
        attribute (str): Attribute to be operated.
        kwargs (dict): Generic parameters.
        dictionary (dict): The default parameters dictionary.
        tab (str): '\t' or '\t\t', depend on orders of attribute.
    Return:
        (str)
    """
    tmp = kwargs.get(attribute) if kwargs.get(attribute) else dictionary.get(attribute).get("value")
    return tab + dictionary.get(attribute).get("name"), str(tmp), dictionary.get(attribute).get("comment")


class MTPotential(Potential):
    """
    This class implements moment tensor potentials.
    """

    pair_style = "pair_style        mlip {}"
    pair_coeff = "pair_coeff        * *"

    def __init__(self, name=None, param=None):
        """

        Args:
            name (str): Name of force field.
            param (dict): The parameter configuration of potentials.
        """
        self.name = name if name else "MTPotential"
        self.mtp_stress_order = ["xx", "yy", "zz", "yz", "xz", "xy"]
        self.vasp_stress_order = ["xx", "yy", "zz", "xy", "yz", "xz"]
        self.param = param if param else None
        self.elements = None

    def _line_up(self, structure, energy, forces, virial_stress):
        """
        Convert input structure, energy, forces, virial_stress to
        proper configuration format for mlip usage.

        Args:
            structure (Structure): Pymatgen Structure object.
            energy (float): DFT-calculated energy of the system.
            forces (list): The forces should have dimension (num_atoms, 3).
            virial_stress (list): stress should has 6 distinct
                elements arranged in order [xx, yy, zz, yz, xz, xy].
        """

        inputs = OrderedDict(
            Size=structure.num_sites,
            SuperCell=structure.lattice,
            AtomData=(structure, forces),
            Energy=energy,
            Stress=virial_stress,
        )

        lines = ["BEGIN_CFG"]

        if "Size" in inputs:
            lines.append(" Size")
            lines.append("{:>7d}".format(inputs["Size"]))
        if "SuperCell" in inputs:
            lines.append(" SuperCell")
            for vec in inputs["SuperCell"].matrix:
                lines.append("{:>17.6f}{:>14.6f}{:>14.6f}".format(*vec))
        if "AtomData" in inputs:
            format_str = "{:>14s}{:>5s}{:>15s}{:>14s}{:>14s}{:>13s}{:>13s}{:>13s}"
            format_float = "{:>14d}{:>5d}{:>15f}{:>14f}{:>14f}{:>13f}{:>13f}{:>13f}"
            lines.append(
                format_str.format("AtomData:  id", "type", "cartes_x", "cartes_y", "cartes_z", "fx", "fy", "fz")
            )
            for i, (site, force) in enumerate(zip(structure, forces)):
                lines.append(format_float.format(i + 1, self.elements.index(str(site.specie)), *site.coords, *force))
        if "Energy" in inputs:
            lines.append(" Energy")
            lines.append("{:>24.12f}".format(inputs["Energy"]))
        if "Stress" in inputs:
            format_str = "{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}"
            format_float = "{:>12f}{:>12f}{:>12f}{:>12f}{:>12f}{:>12f}"
            lines.append(format_str.format("Stress:  xx", "yy", "zz", "yz", "xz", "xy"))
            lines.append(format_float.format(*np.array(virial_stress) / 1.228445))

        lines.append("END_CFG")

        return "\n".join(lines)

    def write_cfg(self, filename, cfg_pool):
        """
        Write configurations to file
        Args:
            filename (str): filename
            cfg_pool (list): list of configurations

        Returns:

        """
        if not self.elements:
            raise ValueError("No species given.")

        lines = []
        for dataset in cfg_pool:
            if isinstance(dataset["structure"], dict):
                structure = Structure.from_dict(dataset["structure"])
            else:
                structure = dataset["structure"]
            energy = dataset["outputs"]["energy"]
            forces = dataset["outputs"]["forces"]
            virial_stress = dataset["outputs"]["virial_stress"]
            virial_stress = [virial_stress[self.vasp_stress_order.index(n)] for n in self.mtp_stress_order]
            lines.append(self._line_up(structure, energy, forces, virial_stress))

        with open(filename, "w") as f:
            f.write("\n".join(lines))

        return filename

    @staticmethod
    def write_ini(Abinitio=0, MLIP="MPT.mpt", Driver=0, **kwargs):
        """
        Write initial file for mlip package.

        Args:
            Abinitio (int): Defines Ab-initio models. Default to 1.

                0: If Ab-initio models is not required.
                1: Used if driver provides EFS data with configurations.
                2: Use embedded Lennard-Jones pair potentials.

                    r_min (float): Distance to minimum of pair function (in Angstroms).
                        Default to 2.0.
                    scale (float): Value of pair function in minimum (in eV).
                        Default to 1.0.
                    cutoff (float): Cutoff radius (in Angstroms). Default to 5.0.

                3: Use DFT models by VASP. Linking via files exchange.

                    POSCAR (str): Relative path of POSCAR file.
                    OUTCAR (str): Relative path of OUTCAR file.
                    Start_command (str): Relative path of command file.

                4: Use potentials calculating by LAMMPS. Linking via files exchange.

                    Input_file (str): File with configuration to be read by lammps.
                    Output_file (str): File with configuration and EFS data to be read by MLIP.
                    Start_command (str): Relative path of command file.

                5: Use MTP as Ab-initio potentials.

                    MTP_filename (str): MTP file name.

            mlip (str): MTP.

                load_from (str): Potential filename.
                Cacluate_EFS (bool): Whether to perform EFS calculation by MTP.
                Fit (bool): Whether to perform MTP learning.

                    Save (str): Output MTP file name (for trained MTP).
                    Energy_equation_weight (float): Weight for energy equation in
                        fitting procedure. Default to 1.0.
                    Forces_equation_weight (float): Weight for forces equations in
                        fitting procedure. Default to 0.001.
                    Stress_equation_weight (float): Weight for stresses equations in
                        fitting procedure.  Default to 0.1.
                    Relative_forces_weight (float): If greater than zero, large forces
                        will be fitted less accurate than small. Default to 0.0.
                    Fit_log (str): File to write fitting log. No logging if not specified.
                        Default to None.

                Select (bool): Whether to activate active learning. Default to False.

                    Site_E_weight (float): Weight for site energy equations in
                        selection procedure. Default to 1.0.
                    Energy_weight (float): Weight for energy equation in
                        selection procedure. Default to 0.0.
                    Forces_weight (float): Weight for forces equations in
                        selection procedure. Default to 0.0.
                    Stress_weight (float): Weight for stresses equations in
                        selection procedure. Default to 0.0.
                    Threshold_slct (float): Selection threshold - maximum
                        allowed extrapolation level. Default to 0.1.
                    Save_TS (str): Filename where selected configurations
                        will be saved. No configuration saving if not specified.
                        Default to None.
                    Save_state (str): Filename where state of the selection
                        will be saved. No saving if not specified. Default to None.
                    Load_state (str): Filename where state of the selection
                        will be loaded. No saving if not specified. Default to None.
                    Select_log (str): File to write fitting log. No logging
                        if not specified. Default to None.

                LOFT (bool): Whether to perform learning on the fly. Default to False

                    EFSviaMTP (bool): Works only on LOFT regime. If True,
                        only MTP-calculated EFS will be passed to driver, else
                        pass to driver ab-initio EFS while LOTF when learning occurs.
                    Log (str): Filename to write log of learning on the fly process.
                        No logging if not specified. Default to None.

                Check_errors (bool): If True, comparison and accumulation of
                    error statistics for EFS calculated by ab-initio models and MTP.
                    Default to False.

                    Log (str): Filename to write log of learning on the fly process.
                        No logging if not specified. Default to None.

                Write_cfgs (bool): File for writing all processed configurations.
                    No confuguration recording if not specified. Default to None.

                    Skip_N (int): The number of configurations to skip while writing.
                        Default to 0.

                Log (str): Filename to write MLIP log. No logging if not specified.
                    Default to None.

            Driver (int): Defines the configuration driver. Default to 1.

                0: No driver or external MD driver.
                1: Read configurations from database file.

                    Database_filename (str): Configuration file name.
                    Max_count (int): Maximal number of configurations to read.
                    Log (str): Filename to write reading log. No logging
                        if not specified. Default to None.

                2: Embedded algorithm for relaxation.

                    Pressure (float): External pressure (in GPa).
                        If not zero enthalpy is minimized. Default to 0.0.
                    Iteration_limit (int): Maximal number of iteration of
                        the relaxation process. Default to 500.
                    Min_dist (float): Minimal interatomic distance constraint
                        (in Angstroms). Default to 1.0.
                    Forces_tolerance (float): Forces on atoms in relaxed
                        configuration should be smaller than this value
                        (in eV/Angstroms). Default to 0.0001.
                    Stress_tolerance (float): Stresses in relaxed configuration
                        should be smaller than this value (in GPa). Default to 0.001.
                    Max_step (float): Maximal allowed displacement of atoms and
                        lattice vectors in Cartesian coordinates (in Angstroms).
                        Default to 0.5.
                    Min_step (float): Minimal displacement of atoms and
                        lattice vectors in Cartesian coordinates (in Angstroms).
                        Default to 1.0e-8.
                    BFGS_Wolfe_C1 (float): Wolfe condition constant on the function
                        decrease (linesearch stopping criterea). Default to 1.0e-3.
                    BFGS_Wolfe_C2 (float): Wolfe condition constant on the gradient
                        decrease (linesearch stopping criterea). Default to 0.7.
                    Save_relaxed (str): Filename for output results of relxation.
                        No configuration will be saved if not specified.
                        Default to None.
                    Log (str): Filename to write relaxation log. No logging
                        if not specified. Default to None.
        """
        lines = []
        format_str = "{:<48s}{:<20s}{}"
        PARAMS = {
            "Abinitio": {
                0: [],
                1: [],
                2: ["r_min", "scale", "cutoff"],
                3: ["POSCAR", "OUTCAR", "Start_command"],
                4: ["Input_file", "Output_file", "Start_command"],
                5: ["MTP_filename"],
            },
            "MLIP": {
                "Calculate_EFS": [],
                "Fit": [
                    "Save",
                    "Energy_equation_weight",
                    "Forces_equation_weight",
                    "Stress_equation_weight",
                    "Relative_forces_weight",
                    "Fit_log",
                ],
                "Select": [
                    "Site_E_weight",
                    "Energy_weight",
                    "Forces_weight",
                    "Stress_weight",
                    "Threshold_slct",
                    "Save_TS",
                    "Save_state",
                    "Load_state",
                    "Select_log",
                ],
                "Write_cfgs": [],
                "Log": [],
            },
            "Driver": {
                0: [],
                1: ["Database_filename", "Database_log"],
                2: [
                    "Pressure",
                    "Iteration_limit",
                    "Min_dist",
                    "Forces_tolerance",
                    "Stress_tolerance",
                    "Max_step",
                    "Min_step",
                    "BFGS_Wolfe_C1",
                    "BFGS_Wolfe_C2",
                    "Save_relaxed",
                    "Relaxation_log",
                ],
            },
        }

        if Abinitio:
            lines.append(
                format_str.format(
                    MTini_params.get("Abinitio").get("name"), str(Abinitio), MTini_params.get("Abinitio").get("comment")
                )
            )
            abinitio = MTini_params.get("Abinitio").get(str(Abinitio))
            lines.append(format_str.format(abinitio.get("name"), "", abinitio.get("comment")))
            for attribute in PARAMS["Abinitio"][Abinitio]:
                lines.append(format_str.format(*feed(attribute, kwargs, abinitio)))

        if MLIP:
            lines.append(
                format_str.format(MTini_params.get("MLIP").get("name"), "mtpr", MTini_params.get("MLIP").get("comment"))
            )
            mlip = MTini_params.get("MLIP")
            if kwargs.get("load_from"):
                load_from = mlip.get("load_from")
                lines.append(
                    format_str.format("\t" + load_from.get("name"), kwargs.get("load_from"), load_from.get("comment"))
                )
            if kwargs.get("Calculate_EFS"):
                calc_efs = mlip.get("Calculate_EFS")
                lines.append(format_str.format("\t" + calc_efs.get("name"), "TRUE", calc_efs.get("comment")))
            if kwargs.get("Fit"):
                fit = mlip.get("Fit")
                lines.append(format_str.format("\t" + fit.get("name"), "true", fit.get("comment")))
                for attribute in PARAMS["MLIP"]["Fit"]:
                    lines.append(format_str.format(*feed(attribute, kwargs, fit, tab="\t\t")))

            if kwargs.get("Select"):
                select = mlip.get("Select")
                lines.append(format_str.format("\t" + select.get("name"), "true", select.get("comment")))
                for attribute in PARAMS["MLIP"]["Select"]:
                    lines.append(format_str.format(*feed(attribute, kwargs, select, tab="\t\t")))

            if kwargs.get("Write_cfgs"):
                write_cfgs = mlip.get("Write_cfgs")
                lines.append(
                    format_str.format(
                        "\t" + write_cfgs.get("name"), kwargs.get("Write_cfgs"), write_cfgs.get("comment")
                    )
                )

        if Driver:
            lines.append(
                format_str.format(
                    MTini_params.get("Driver").get("name"), str(Driver), MTini_params.get("Driver").get("comment")
                )
            )
            driver = MTini_params.get("Driver").get(str(Driver))
            lines.append(format_str.format(driver.get("name"), "", driver.get("comment")))
            for attribute in PARAMS["Driver"][Driver]:
                lines.append(format_str.format(*feed(attribute, kwargs, driver, tab="\t\t")))

        filename = "mlip.ini"
        with open(filename, "w") as f:
            f.write("\n".join(lines))

        return filename

    def read_cfgs(self, filename):
        """

        Args:
            filename (str): The configuration file to be read.

        """

        def formatify(string):
            return [float(s) for s in string.split()]

        if not self.elements:
            raise ValueError("No species given.")

        data_pool = []
        with zopen(filename, "rt") as f:
            lines = f.read()

        block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
        size_pattern = re.compile("Size\n(.*?)\n SuperCell", re.S | re.I)
        lattice_pattern = re.compile("SuperCell\n(.*?)\n AtomData", re.S | re.I)
        position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
        energy_pattern = re.compile("Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
        stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)
        for block in block_pattern.findall(lines):
            d = {"outputs": {}}
            size_str = size_pattern.findall(block)[0]
            size = int(size_str.lstrip())
            lattice_str = lattice_pattern.findall(block)[0]
            lattice = Lattice(np.array(list(map(formatify, lattice_str.split("\n")))))
            position_str = position_pattern.findall(block)[0]
            position = np.array(list(map(formatify, position_str.split("\n"))))
            species = np.array(self.elements)[position[:, 1].astype(np.int)]
            forces = position[:, 5:8].tolist()
            position = position[:, 2:5]
            energy_str = energy_pattern.findall(block)[0]
            energy = float(energy_str.lstrip())
            stress_str = stress_pattern.findall(block)[0]
            virial_stress = (
                np.array(list(map(formatify, stress_str.split())))
                .reshape(
                    6,
                )
                .tolist()
            )
            virial_stress = [virial_stress[self.mtp_stress_order.index(n)] for n in self.vasp_stress_order]
            struct = Structure(lattice=lattice, species=species, coords=position, coords_are_cartesian=True)
            d["structure"] = struct.as_dict()
            d["outputs"]["energy"] = energy
            assert size == struct.num_sites
            d["num_atoms"] = size
            d["outputs"]["forces"] = forces
            d["outputs"]["virial_stress"] = virial_stress

            data_pool.append(d)
        _, df = convert_docs(docs=data_pool)
        return data_pool, df

    def train(
        self,
        train_structures,
        train_energies,
        train_forces,
        train_stresses=None,
        unfitted_mtp=None,
        max_dist=5,
        radial_basis_size=8,
        max_iter=500,
        energy_weight=1,
        force_weight=1e-2,
        stress_weight=0,
    ):
        """
        Training data with moment tensor method.

        Args:
            train_structures ([Structure]): The list of Pymatgen Structure object.
                energies ([float]): The list of total energies of each structure
                in structures list.
            train_energies ([float]): List of total energies of each structure in
                structures list.
            train_forces ([np.array]): List of (m, 3) forces array of each structure
                with m atoms in structures list. m can be varied with each single
                structure case.
            train_stresses (list): List of (6, ) virial stresses of each structure
                in structures list.
            unfitted_mtp (str): Define the initial mtp file. Default to the mtp file
                stored in .params directory.
            max_dist (float): The actual radial cutoff.
            radial_basis_size (int): Relevant to number of radial basis function.
            max_iter (int): The number of maximum iteration.
            energy_weight (float): The weight of energy.
            force_weight (float): The weight of forces.
            stress_weight (float): The weight of stresses.
        """
        if not which("mlp"):
            raise RuntimeError(
                "mlp has not been found.\n",
                "Please refer to http://gitlab.skoltech.ru/shapeev/mlip ",
                "for further detail.",
            )
        train_structures, train_forces, train_stresses = check_structures_forces_stresses(
            train_structures, train_forces, train_stresses
        )
        train_pool = pool_from(train_structures, train_energies, train_forces, train_stresses)
        elements = sorted(set(itertools.chain(*[struct.species for struct in train_structures])))
        self.elements = [str(element) for element in elements]

        atoms_filename = "train.cfgs"

        with ScratchDir("."):
            atoms_filename = self.write_cfg(filename=atoms_filename, cfg_pool=train_pool)

            if not unfitted_mtp:
                raise RuntimeError("No specific parameter file provided.")
            MTP_file_path = os.path.join(module_dir, "params", unfitted_mtp)
            shutil.copyfile(MTP_file_path, os.path.join(os.getcwd(), unfitted_mtp))

            with open("min_dist", "w") as f:
                p = subprocess.Popen(["mlp", "mindist", atoms_filename], stdout=f)
            stdout = p.communicate()[0]

            with open("min_dist") as f:
                lines = f.readlines()
            min_dist = float(lines[-1].split(":")[1])

            with open(unfitted_mtp) as f:
                template = f.read()

            s = template % (len(self.elements), min_dist, max_dist, radial_basis_size)
            with open(unfitted_mtp, "w") as f:
                f.write(s)

            save_fitted_mtp = ".".join([unfitted_mtp.split(".")[0] + "_fitted", unfitted_mtp.split(".")[1]])

            p = subprocess.Popen(
                [
                    "mlp",
                    "train",
                    unfitted_mtp,
                    atoms_filename,
                    "--max-iter={}".format(max_iter),
                    "--trained-pot-name={}".format(save_fitted_mtp),
                    "--curr-pot-name={}".format(unfitted_mtp),
                    "--energy-weight={}".format(energy_weight),
                    "--force-weight={}".format(force_weight),
                    "--stress-weight={}".format(stress_weight),
                    "--init-params=same",
                    "--auto-min-dist",
                ],
                stdout=subprocess.PIPE,
            )
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = "MLP exited with return code %d" % rc
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg) if m.startswith("ERROR")][0]
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            def load_config(filename):
                param = OrderedDict()
                with open(filename, "r") as f:
                    lines = f.readlines()
                param["safe"] = [line.rstrip() for line in lines[:-2]]
                for line in lines[-2:]:
                    key = line.rstrip().split(" = ")[0]
                    value = json.loads(line.rstrip().split(" = ")[1].replace("{", "[").replace("}", "]"))
                    param[key] = value
                return param

            self.param = load_config(save_fitted_mtp)
        return rc

    def write_param(self, fitted_mtp="fitted.mtp", **kwargs):
        """
        Write fitted mtp parameter file to perform lammps calculation.

        Args:
            fitted_mtp (str): Filename to store xml formatted parameters.
        """
        if not self.param:
            raise RuntimeError("The parameters should be provided.")
        lines = [
            " = ".join([key, json.dumps(value).replace("[", "{").replace("]", "}")])
            if key != "safe"
            else "\n".join(value)
            for key, value in self.param.items()
        ]
        with open(fitted_mtp, "w") as f:
            f.write("\n".join(lines))
        ini_file = self.write_ini(load_from=fitted_mtp, Calculate_EFS=True, **kwargs)
        ff_settings = [self.pair_style.format(ini_file), self.pair_coeff]
        return ff_settings

    def evaluate(self, test_structures, test_energies, test_forces, test_stresses=None, **kwargs):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures ([Structure]): List of Pymatgen Structure Objects.
            test_energies ([float]): List of DFT-calculated total energies of
                each structure in structures list.
            test_forces ([np.array]): List of DFT-calculated (m, 3) forces of
                each structure with m atoms in structures list. m can be varied
                with each single structure case.
            test_stresses (list): List of DFT-calculated (6, ) viriral stresses
                of each structure in structures list.
            kwargs: Parameters of write_param method.
        """
        if not which("mlp"):
            raise RuntimeError(
                "mlp has not been found.\n",
                "Please refer to http://gitlab.skoltech.ru/shapeev/mlip ",
                "for further detail.",
            )
        fitted_mtp = "fitted.mtp"
        original_file = "original.cfgs"
        predict_file = "predict.cfgs"
        test_structures, test_forces, test_stresses = check_structures_forces_stresses(
            test_structures, test_forces, test_stresses
        )
        predict_pool = pool_from(test_structures, test_energies, test_forces, test_stresses)

        with ScratchDir("."):
            self.write_param(
                fitted_mtp=fitted_mtp,
                Abinitio=0,
                Driver=1,
                Write_cfgs=predict_file,
                Database_filename=original_file,
                **kwargs,
            )
            original_file = self.write_cfg(original_file, cfg_pool=predict_pool)
            _, df_orig = self.read_cfgs(original_file)

            p = subprocess.Popen(
                ["mlp", "run", "mlip.ini", "--filename={}".format(original_file)], stdout=subprocess.PIPE
            )
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = "mlp exited with return code %d" % rc
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg) if m.startswith("ERROR")][0]
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            if not os.path.exists(predict_file):
                predict_file = "_".join([predict_file, "0"])
            _, df_predict = self.read_cfgs(predict_file)
        return df_orig, df_predict

    def predict_efs(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        calculator = EnergyForceStress(self)
        energy, forces, stress = calculator.calculate(structures=[structure])[0]
        return energy, forces, stress

    @staticmethod
    def from_config(filename, elements):
        """
        Initialize potentials with parameters file.

        Args:
            filename (str): The file storing parameters of potentials, filename should
                ends with ".mtp".
            elements (list): The list of elements.

        Returns:
            MTPotential
        """
        param = OrderedDict()
        with open(filename, "r") as f:
            lines = f.readlines()
        param["safe"] = [line.rstrip() for line in lines[:-2]]
        for line in lines[-2:]:
            key = line.rstrip().split(" = ")[0]
            value = json.loads(line.rstrip().split(" = ")[1].replace("{", "[").replace("}", "]"))
            param[key] = value

        mtp = MTPotential(param=param)
        mtp.elements = elements

        return mtp
