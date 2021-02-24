"""
Class for performing local environment predictions
"""

import json
import logging
import os
import warnings
from numbers import Number
from typing import List, Optional, Union
from zipfile import ZipFile
import requests
import numpy as np

import joblib

from scipy.interpolate import interp1d
from tqdm import tqdm

from maml.apps.rfxas._core import XANES


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CWD = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(CWD, "data")
EXTRA_MODELS = os.path.join(CWD, "ex_models.zip")
EXTRA_MODEL_DIR = os.path.join(CWD, "ex_models/Model_share")
EXTRA_MODEL_SUB_DICT = {"cnn": "CNN_model", "knn": "KNN_model", "mlp": "MLP_model", "svc": "SVC_model"}
EXTRA_TEMPLATES = {"cnum": "{}_c_num", "cmotif": "{}_c_env_{}_env_label"}

EXTRA_MODEL_URL = "https://ndownloader.figshare.com/files/18741755"
RF_MODEL_PATH = os.path.join(CWD, "models.zip")
RF_MODEL_URL = "https://ndownloader.figshare.com/files/25946756"


CN_CLASSES = [
    "CN_1-CN_4",
    "CN_12",
    "CN_12-CN_10",
    "CN_12-CN_3",
    "CN_12-CN_6-CN_9",
    "CN_2",
    "CN_2-CN_4-CN_6",
    "CN_2-CN_6",
    "CN_2-CN_6-CN_5",
    "CN_3",
    "CN_3-CN_4",
    "CN_3-CN_5",
    "CN_3-CN_5-CN_6",
    "CN_3-CN_6",
    "CN_3-CN_6-CN_4",
    "CN_4",
    "CN_4-CN_1",
    "CN_4-CN_2",
    "CN_4-CN_3",
    "CN_4-CN_3-CN_5-CN_6",
    "CN_4-CN_3-CN_6",
    "CN_4-CN_5",
    "CN_4-CN_5-CN_6",
    "CN_4-CN_6",
    "CN_4-CN_6-CN_5",
    "CN_4-CN_8",
    "CN_5",
    "CN_5-CN_3",
    "CN_5-CN_3-CN_4",
    "CN_5-CN_3-CN_6",
    "CN_5-CN_4",
    "CN_5-CN_4-CN_6",
    "CN_5-CN_6",
    "CN_5-CN_6-CN_3",
    "CN_5-CN_6-CN_7",
    "CN_5-CN_7-CN_6",
    "CN_6",
    "CN_6-CN_1",
    "CN_6-CN_12",
    "CN_6-CN_2",
    "CN_6-CN_3",
    "CN_6-CN_3-CN_5",
    "CN_6-CN_4",
    "CN_6-CN_4-CN_5",
    "CN_6-CN_5",
    "CN_6-CN_5-CN_3",
    "CN_6-CN_7",
    "CN_6-CN_8",
    "CN_6-CN_8-CN_4",
    "CN_7",
    "CN_7-CN_6",
    "CN_7-CN_9",
    "CN_8",
    "CN_8-CN_7",
]


with open(os.path.join(DATA_DIR, "cmotif_labels.json"), "r") as f:
    CMOTIF_LABELS = json.load(f)


class CenvPrediction:
    """
    Coordination environment predictor. Supported models are
        RandomForest (rf)
        K-neareset Neighbors (knn)
        Support-vector Classifier (svc)
        Multi-layer Perceptron (mlp)
        Convolutional Neural Network (cnn)
    """

    _available_models = ["rf", "knn", "svc", "mlp", "cnn"]

    def __init__(
        self,
        xanes_spectrum: XANES,
        energy_reference: str,
        energy_range: Union[float, List[float]],
        edge_energy: Optional[float] = None,
        spectrum_interpolation: bool = True,
        model: str = "rf",
        model_dir: Optional[str] = None,
        extra_model_dir: str = EXTRA_MODEL_DIR,
    ):
        """
        Args:
            xanes_spectrum: XANES object
            energy_reference (str): Energy reference mode, choose from 'lowest' or 'E0'. 'lowest' mode for
                            using the lowest energy of spectrum as the starting point of the to be characterized
                            spectrum. 'E0' mode for using the edge energy as the reference point to generate to be
                            characterized spectrum energy range.
            energy_range (list/float): Energy range of spectrum used for prediction. If the energy reference mode is
                'lowest', energy range value need to be a number specifies the energy range. If the energy
                reference mode is 'E0', energy range ought to be a list of two numbers. The first number
                (negative) represents the difference between the lower bound energy and energy reference.
            edge_energy (float): Edge energy of spectra. Usually determined using MBACK algorithm and provided by users
            spectrum_interpolation: Whether or not do spectrum interpolation. Default to True. If
                spectrum_interpolation option if False, then the CenvPrediction object will use the original spectrum
                of xanes_spectrum for coordination environment prediction. The original spectrum need to be a vector
                with length equals 200.
            model (str): rf, knn, mlp or svc. Default is rf. For other models, models download ~1.28 GB is required.
            model_dir (str): specifies where the models random forest models are stored.
                if the dir does not exists, the models will be downloaded from figshare
            extra_model_dir (str): specifies where the extra models are stored.
                if the dir does not exists, the extra models will be downloaded from figshare
        """

        self.xanes_spectrum = xanes_spectrum
        self.absorption_specie = self.xanes_spectrum.absorption_specie
        self.energy_reference = energy_reference
        self.energy_range = energy_range
        self.cnum_model_name_template = "RandomForest_{}_c_num.sav"
        self.cmotif_model_name_template = "RandomForest_{}_c_env_ex_{}.sav"
        if model_dir is None:
            model_dir = os.path.join(CWD, "models")
        self.model_dir = model_dir

        if not os.path.isdir(self.model_dir):
            _download_models(RF_MODEL_URL, RF_MODEL_PATH, self.model_dir)

        if model not in self._available_models:
            raise ValueError("Model type %s not recognized " % str(model), " choose from ", self._available_models)
        if model != "rf":
            if os.path.isdir(extra_model_dir):
                pass
            else:
                _download_models(EXTRA_MODEL_URL, EXTRA_MODELS, extra_model_dir)

            self.model_dir = os.path.join(extra_model_dir, EXTRA_MODEL_SUB_DICT[model])

            if model.lower() == "cnn":
                suffix = ".h5"
            else:
                suffix = ".sav"
            self.cnum_model_name_template = model.upper() + "_" + EXTRA_TEMPLATES["cnum"] + suffix
            self.cmotif_model_name_template = model.upper() + "_" + EXTRA_TEMPLATES["cmotif"] + suffix

        if isinstance(self.energy_range, list):
            self.energy_lower_bound = self.energy_range[0]
            self.energy_higher_bound = self.energy_range[-1]

        self._parameter_validation()

        if energy_reference == "E0" and edge_energy is None:
            warnings.warn(
                "Using edge energy of xanes_spectrum object, be cautious about how the object's edge energy"
                " is determined"
            )
            self.edge_energy = self.xanes_spectrum.e0
        elif energy_reference == "E0" and edge_energy:
            self.edge_energy = edge_energy
        elif self.energy_reference == "lowest":
            self.edge_energy = self.xanes_spectrum.e0

        if spectrum_interpolation:
            self._energy_interp()
        else:
            self.interp_spectrum = self.xanes_spectrum.y
            self.interp_spectrum_reshape = np.array(self.interp_spectrum).reshape(1, -1)
            self.interp_energy = self.xanes_spectrum.x

        self.model = model.lower()
        if self.model == "cnn":
            self.interp_spectrum_reshape = self.interp_spectrum_reshape.reshape((1, -1, 1))

    def cenv_prediction(self):
        """
        Function used for coordination environment prediction

        Returns:
            Two new attributes of CenvPrediction object. The attribute 'pred_cnum_ranklist' is the predicted
            coordination number ranklist. The attribute pred_cenv contains the final predicted coordination
            environment ranklists.

        """
        self._cnum_prediction()
        self._cmotif_prediction()

    def _cnum_prediction(self):
        cnum_pred_ele_json = os.path.join(DATA_DIR, "cnum_predict_elements.json")
        with open(cnum_pred_ele_json, "r") as fp:
            cnum_pred_elements = json.load(fp)

        if self.absorption_specie not in cnum_pred_elements:
            warning_msg = "Coordination number prediction models for {} is unavailable.".format(self.absorption_specie)
            warnings.warn(warning_msg)
            self.pred_cnum_ranklist = "cnum undetermined"
        else:
            cnum_model_name = self.cnum_model_name_template.format(self.absorption_specie)
            cnum_model_path = os.path.join(self.model_dir, "cnum", cnum_model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                logger.info("Loaded %s models" % cnum_model_path)
                if cnum_model_path.endswith(".sav"):
                    cnum_model_loaded = joblib.load(cnum_model_path)
                elif cnum_model_path.endswith(".h5"):
                    import tensorflow as tf

                    print("cnum ", cnum_model_path)
                    cnum_model_loaded = tf.keras.models.load_model(cnum_model_path)
                else:
                    raise ValueError("Cnum models not recognized")
            self.pred_cnum_ranklist = cnum_model_loaded.predict(self.interp_spectrum_reshape)
            if self.model == "cnn":
                self.pred_cnum_ranklist = CN_CLASSES[np.argmax(self.pred_cnum_ranklist)]
            else:
                self.pred_cnum_ranklist = self.pred_cnum_ranklist[0]

    def _cmotif_prediction(self):
        cmotif_pred_ele_json = os.path.join(DATA_DIR, "cmotif_predict_elements.json")
        with open(cmotif_pred_ele_json, "r") as fp:
            cmotif_ele_env_dict = json.load(fp)
        ele_env_valid_prediction = cmotif_ele_env_dict[self.absorption_specie]

        if self.pred_cnum_ranklist == "cnum undetermined":
            self.pred_cenv = "cenv undetermined"
        else:
            pred_cnum_ranklist = self.pred_cnum_ranklist
            # Using predicted cnum ranklist to predict cenv ranklist
            spectral_env_pred = []
            for indi_pred_cnum in pred_cnum_ranklist.split("-"):
                cmotif_pred_cenv = "ex_{}".format(indi_pred_cnum)

                # No available coord. motif prediction models for this particular coord. num. of this element
                if cmotif_pred_cenv not in ele_env_valid_prediction:
                    pseudo_cmotif_label = "{} coord. motif undetermined".format(indi_pred_cnum)
                    spectral_env_pred.append(pseudo_cmotif_label)
                else:
                    cmotif_model_name = self.cmotif_model_name_template.format(self.absorption_specie, indi_pred_cnum)
                    cmotif_model_path = os.path.join(self.model_dir, "cmotif", cmotif_model_name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        logger.info("Loaded %s models" % cmotif_model_path)
                        if cmotif_model_path.endswith(".sav"):
                            cmotif_model_loaded = joblib.load(cmotif_model_path)
                        elif cmotif_model_path.endswith(".h5"):
                            import tensorflow as tf

                            print("cmotif", cmotif_model_path)
                            cmotif_model_loaded = tf.keras.models.load_model(cmotif_model_path)
                        else:
                            raise ValueError("Model format not recognized")

                        if self.model == "cnn":
                            pred_motif_ranklist = cmotif_model_loaded.predict(
                                self.interp_spectrum_reshape.reshape((1, -1, 1))
                            )
                            labels = CMOTIF_LABELS[cmotif_model_name[4:-3]]
                            pred_motif_ranklist = labels[np.argmax(pred_motif_ranklist)]
                        else:
                            pred_motif_ranklist = cmotif_model_loaded.predict(self.interp_spectrum_reshape)[0]
                        pred_cnum_cmotif_concat = "-".join([indi_pred_cnum, pred_motif_ranklist])
                        spectral_env_pred.append(pred_cnum_cmotif_concat)

            self.pred_cenv = spectral_env_pred

    def _energy_interp(self):
        # if energy_reference is 'lowest' and energy range is proper passed
        if self.energy_reference == "lowest":
            x_axis_energy = self.xanes_spectrum.x
            y_spectrum = self.xanes_spectrum.y
            x_axis_energy_start = np.min(x_axis_energy)
            x_axis_energy_end = x_axis_energy_start + self.energy_range
            x_axis_energy_end_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_end)
            x_axis_linspace = np.linspace(x_axis_energy_start, x_axis_energy[x_axis_energy_end_index], num=200)
            f = interp1d(x_axis_energy, y_spectrum, kind="cubic", bounds_error=False, fill_value=0)
            self.interp_spectrum = f(x_axis_linspace)
            normalize_factor = np.max(self.interp_spectrum, axis=0)
            self.interp_spectrum /= normalize_factor
            self.interp_energy = x_axis_linspace
            self.interp_spectrum_reshape = np.array(self.interp_spectrum).reshape(1, -1)

        elif self.energy_reference == "E0":
            x_axis_energy = self.xanes_spectrum.x
            y_spectrum = self.xanes_spectrum.y
            x_axis_energy_start = self.edge_energy + self.energy_lower_bound
            x_energy_start_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_start)
            x_axis_energy_end = self.edge_energy + self.energy_higher_bound
            x_energy_end_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_end)
            x_axis_linspace = np.linspace(
                x_axis_energy[x_energy_start_index], x_axis_energy[x_energy_end_index], num=200
            )
            f = interp1d(x_axis_energy, y_spectrum, kind="cubic", bounds_error=False, fill_value=0)
            self.interp_spectrum = f(x_axis_linspace)
            normalize_factor = np.max(self.interp_spectrum, axis=0)
            self.interp_spectrum /= normalize_factor
            self.interp_spectrum_reshape = np.array(self.interp_spectrum).reshape(1, -1)
            self.interp_energy = x_axis_linspace

    def _parameter_validation(self):
        if self.energy_reference not in ["lowest", "E0"]:
            raise ValueError('Invalid energy reference option, energy_reference should either be "lowest" or "E0"')

        if self.energy_reference == "lowest":
            if not isinstance(self.energy_range, Number):
                raise ValueError(
                    "Energy range needs to be a number when the energy reference point is the starting energy of "
                    "the spectrum"
                )

            if self.energy_range < 0.0:  # type: ignore
                raise ValueError("Energy range needs to be larger than 0. Invalid energy range error.")

        if self.energy_reference == "E0":
            if not isinstance(self.energy_range, list):
                raise ValueError(
                    "Energy range needs to be a list contains lower energy bound and higher energy bound refer to "
                    "energy reference point"
                )

            if self.energy_lower_bound > 0:
                raise ValueError("Energy lower bound needs to be less than zero.")

            if self.energy_higher_bound < 0:
                raise ValueError("Energy higher bound needs to be larger than zero.")


def find_nearest_energy_index(energy_array, energy_value):
    """
    Given a target energy value, returns a value index of energy_array with index value most close to the target energy
    value

    Args:
        energy_array: Energy array to search for the closest energy value index
        energy_value: Target energy value for index searching.

    Returns:

    """
    energy_array = np.asarray(energy_array)
    energy_index = (np.abs(energy_array - energy_value)).argmin()
    return energy_index


def _download_models(url, file_path=EXTRA_MODELS, dest="ext_models"):
    """
    Download machine learning models files

    Args:
        url: (str) url link for the models
        dest: (str) destination for extraction
    """

    logger.info("Fetching {} from {} to {}".format(os.path.basename(file_path), url, file_path))
    if not os.path.isfile(file_path):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 Mbyte
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(file_path, "wb") as file_out:
            for chunk in r.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                file_out.write(chunk)
        t.close()
        r.close()
    logger.info("Start extracting models to %s ..." % dest)
    with ZipFile(file_path, "r") as zip_obj:
        zip_obj.extractall(dest)
