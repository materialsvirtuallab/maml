"""
MamlDB controls the database operations for maml
"""


from typing import Dict, List, Optional
import warnings

from pymongo import MongoClient
from monty.json import MontyDecoder
from tqdm import tqdm

from maml.base import BaseModel, BaseDescriber
from maml.utils import DataSplitter, ShuffleSplitter


class LockedError(ValueError):
    """
    Operations on one document should be mutually exclusive at any given moment
    """
    pass


class MamlDB:
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 27017,
                 name: str = "maml",
                 username: str = None,
                 password: str = None,
                 ssl: bool = False,
                 ssl_ca_certs: str = None,
                 ssl_certfile: str = None,
                 ssl_keyfile: str = None,
                 ssl_pem_passphrase: str = None,
                 uri_mode: bool = False,
                 mongoclient_kwargs: Dict = None):
        """
        Args:
            host (str): hostname. If uri_mode is True, a MongoDB connection string URI
                (https://docs.mongodb.com/manual/reference/connection-string/) can be used instead of the remaining
                options below.
            port (int): port number
            name (str): database name
            username (str)
            password (str)
            ssl (bool): use TLS/SSL for mongodb connection
            ssl_ca_certs (str): path to the CA certificate to be used for mongodb connection
            ssl_certfile (str): path to the client certificate to be used for mongodb connection
            ssl_keyfile (str): path to the client private key
            ssl_pem_passphrase (str): passphrase for the client private key
            uri_mode (bool): if set True, all Mongo connection parameters occur through a MongoDB URI string (set as
                the host).
            mongoclient_kwargs (dict): A list of any other custom keyword arguments to be
                passed into the MongoClient connection (non-URI mode only)
        """
        self.host = host
        self.port = port
        self.name = name
        self.username = username
        self.password = password
        self.ssl = ssl
        self.ssl_ca_certs = ssl_ca_certs
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_pem_passphrase = ssl_pem_passphrase
        self.mongoclient_kwargs = mongoclient_kwargs or {}
        self.uri_mode = uri_mode

        if self.uri_mode:
            self.connection = MongoClient(host)
            dbname = host.split('/')[-1].split('?')[0]
            self.db = self.connection[dbname]

        else:
            self.connection = MongoClient(self.host, self.port, ssl=self.ssl,
                                          ssl_ca_certs=self.ssl_ca_certs,
                                          ssl_certfile=self.ssl_certfile,
                                          ssl_keyfile=self.ssl_keyfile,
                                          ssl_pem_passphrase=self.ssl_pem_passphrase,
                                          **self.mongoclient_kwargs)

            self.db = self.connection[self.name]
            self.db.authenticate(self.username, self.password)
        self.data_info = self.db.data_info
        self.data = self.db.data
        self.data_split = self.db.data_split
        self.model = self.db.model
        self.model_results = self.db.model_results
        self.descriptor_info = self.db.descriptor_info
        self.descriptors = self.db.descriptors

    def insert_data(self,
                    data_dict_list: List[Dict],
                    data_name: str,
                    input: str,
                    input_type: str,
                    target: str,
                    target_unit: str,
                    task_type: str = 'regression'):
        """
        Insert data to the data collection and update the data_info collections

        Args:
            data_dict_list (list): a list of data docs
            data_name (str): data name
            input (str): tag for data input, e.g., structure or composition
            input_type (str): input types, e.g., pymatgen.core.structure.Structure
            target (str): target name in the doc, e.g.,, eform, bandgap
            target_unit (str): unit for target, e.g., eV/atom
            task_type (str): task type, e.g., classification, regression

        """

        if not self._check_data(data_dict_list, input, target):
            raise ValueError('Not all data has the set input and target')

        if not all(['mat_id' in d for d in data_dict_list]):
            for i, d in enumerate(data_dict_list):
                d['mat_id'] = i

        self.data_info.insert_one({'data_name': data_name,
                                   'input': input,
                                   'input_type': input_type,
                                   'target':  target,
                                   'target_unit': target_unit,
                                   'task_type': task_type})
        self.data.insert_many(d)
        self._update_data_counts(data_name)

    def _update_data_counts(self, data_name: Optional[str] = None):
        """
        Update data counts in data_info collection

        """
        if data_name is None:
            counts = list(self.data.aggregate([
                {"$group": {"_id": "$data_name", "count": {"$sum": 1}}}]))
            for count in counts:
                self.data_info.update_one({'data_name': count['_id']},
                                          {"$set": {"count": count['count']}})
        else:
            count = self.data.find({'data_name': data_name}).count()
            self.data_info.update_one({'data_name': data_name},
                                      {"$set": {"count": count}})

    def _check_data(self, data_dict_list: List[Dict],
                    input: str, target: str) -> bool:
        """
        Check if all data doc contains input and target tags
        Args:
            data_dict_list (list): list of data docs
            input (str): input name
            target (str): target name

        Returns: whether to data list pass

        """
        return all([_dict_contain(i, [input, target]) for i in data_dict_list])

    def generate_split(self,
                       data_name: str,
                       split_name: Optional[str] = None,
                       splitter: Optional[DataSplitter] = None,
                       with_input: bool = False,
                       with_target: bool = False):
        """
        Generate split
        Args:
            data_name (str): the data name to split
            split_name (str): the unique name for split
            splitter (DataSplitter): split
            with_input (bool): whether the splitter needs
                input information
            with_target (bool): whether the splitter needs
                target information
        Returns: split name str
        """
        if splitter is None:
            splitter = ShuffleSplitter()
        projection = []
        if with_input:
            projection.append('input')
        if with_target:
            projection.append('target')
        # get input and target
        input_target = self.data_info.find_one(
            {'data_name': data_name}, projection=projection)

        if len(input_target) == 0:
            return

        keys = list(input_target.keys())
        keys += ['mat_id']
        data = list(self.data.find({'data_name': data_name}, projection=keys))
        keys.remove('mat_id')
        mat_ids = [i['mat_id'] for i in data]

        info = {i: [j[input_target[i]] for j in data] for i in projection}
        split_ids = splitter.split(mat_ids, **info)
        if len(split_ids) == 2:
            id_keys = ['train_ids', 'test_ids']
        elif len(split_ids) == 3:
            id_keys = ['train_ids', 'val_ids', 'test_ids']
        else:
            raise ValueError('Only 2 or 3 splits are accepted for mat ids')

        splits = {}
        for i, j in zip(id_keys, split_ids):
            splits[i] = j

        if split_name is None:
            split_name = '%s-%s' % (data_name,
                                    splitter.__class__.__name__)

        names = [i['split_name'] for i in
                 self.data_split.find({}, projection=['split_name'])
                 if split_name in i['split_name']]

        if len(names) == 0:
            ind = 0
        else:
            ind = max([int(i.split('-')[-1]) for i in names]) + 1

        splits['split_name'] = '%s-index-%d' % (split_name, ind)
        splits['data_name'] = data_name
        # see if the split is already there

        self.data_split.insert_one(splits)
        return splits['split_name']

    def delete_split(self, split_name):
        """
        Delete specific split using its name
        Args:
            split_name (str): unique name for the split

        Returns:

        """
        self.data_split.delete_one({'split_name': split_name})

    def get_available_data_names(self) -> List[str]:
        """
        Get the unique data_name in the data_info collection

        Returns: list of data names

        """
        return [i['data_name']
                for i in self.data_info.find({}, projection=['data_name'])]

    def summarize_data(self, data_name: str = None) -> str:
        """
        Summarize available data in the db.
        Format will be as follows for example

        mp_formation has a total of 100000 (structure, formation_energy_per_atom)
        pairs, and it has [split_1, split_2, split_3] splits with train, val and test
        sizes in each splits as follows:
            - split_1, 80%-10%-10%, generated by random sampling
            - split_2, 20%-40%-40%, generated by stratefied sampling
            ...
        The descriptors calculated on the mp_formation data are the follows
            - BispectrumCoefficients with parameters ...
            - MEGNetElement with parameters ..


        Args:
            data_name (str): data name for the summary, if None, summarize all data
        Returns:

        """
        data_names = self.get_available_data_names()
        if data_name not in data_names:
            return

    def insert_descriptors(self, data_name: str, describer: BaseDescriber,
                           unique_name: str, verbose: bool = True):
        """
        Insert descriptors for data_name
        Args:
            data_name:
            describer:
            unique_name:
            verbose:
        Returns:
        """
        md = MontyDecoder()

        data_info = self.data_info.find_one({'data_name': data_name})
        data_query = list(self.data.find({'data_name': data_name}))

        inp = data_info['input']
        target = data_info['target']

        mat_ids = [i['mat_id'] for i in data_query]
        objs = [md.process_decoded(i[inp]) for i in data_query]
        targets = [i[target] for i in data_query]

        if verbose:
            objs = tqdm(objs)

        final_mat_ids = []
        features = []
        final_targets = []
        for mat_id, obj, target in zip(mat_ids, objs, targets):
            try:
                f = describer.transform_one(obj).values.tolist()
                features.append(f)
                final_mat_ids.append(mat_id)
                final_targets.append(target)

            except Exception as e:
                warnings.warn("Obj %s calculation failed, with error %s"
                              % (str(obj), str(e)))

        docs = [{'data_name': data_name,
                 'descriptor_name': unique_name,
                 'descriptor': f,
                 'target': t,
                 'mat_id': mat_id} for mat_id, t, f in zip(final_mat_ids,
                                                           final_targets,
                                                           features)]
        self.descriptor_info.insert_one({'descriptor_name': unique_name,
                                         'descriptor_dict': describer.as_dict()})

        self.descriptors.insert_many(docs)

    def summarize_model(self) -> str:
        pass

    def insert_model(self, model: BaseModel, model_name: str):
        """
        Insert a model into the database

        Args:
            model (BaseModel): a maml model
            model_name (str): name identifier

        Returns: None

        """
        pass

    @classmethod
    def from_file(cls, filename: str, admin: bool = False):
        import json
        with open(filename, 'r') as f:
            d = json.load(f)

        settings = {}

        for key in ['host', 'port']:
            settings[key] = d[key]

        settings['username'] = d["admin_user"] if admin else d['readonly_user']
        settings['password'] = d["admin_password"] if admin else d['readonly_password']
        settings['name'] = d["database"]
        return cls(**settings)


def _dict_contain(d: Dict, keys: List[str]) -> bool:
    return all([i in d for i in keys])

