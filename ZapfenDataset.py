import torch
import os
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
from sci_analysis import analyze


class ZapfenDataset(Dataset):

    def __init__(self, csv_file):
        """Loads the samples in the csv_file.

        Args:
            csv_file (str): Location of csv file.
        """
        self._setup_logging()
        self.label_distribution = {'P.m.ssp.m.': 0, 'P.s.': 0, 'P.s.xu.': 0, 'P.m.ssp.u.': 0}
        # open csv file
        with open(csv_file, mode='r') as csv_file:
            dict_reader = csv.DictReader(csv_file, delimiter=';')

            features, labels = [], []
            features_invalid, labels_invalid = [], []
            num_invalid_rows = 0

            # iterate over rows
            for row in dict_reader:

                # check if row has invalid values
                row_is_valid = True
                if 'NA' in row.values():
                    row_is_valid = False
                    # put magic value
                    for key in row.keys():
                        if row[key] == 'NA':
                            row[key] = -1.0

                # get useful feature values
                extracted_row = \
                    [self._map_oeffnung(row['Z_Oeffnung']), row['Z_laeng'], row['Z_breit'],
                        row['DM_Spitz'], row['DM_max'], row['DM_mit'], row['Woel_aufg'],
                        row['Woel_flach'], row['Verh_LB'], row['Asym'], row['Apo_L'],
                        row['Apo_B'], row['Apo_S'], row['Entf'], row['Verh_LABA'], row['Verh_LASA'],
                        row['Hakigkeit'], row['Stiel_L']]
                # string to float
                extracted_row = [float(e.replace(',', '.')) if type(e) is str else e for e in extracted_row ]

                # filter for NA values
                if row_is_valid:
                    features.append(extracted_row)
                    labels.append(self._map_label(row['Bart']))
                else:
                    num_invalid_rows += 1
                    features_invalid.append(extracted_row)
                    labels_invalid.append(self._map_label(row['Bart']))

            # convert to pytorch datatype
            s = np.array(features).shape
            s_2 = np.array(features_invalid).shape
            self.features = torch.FloatTensor(features)
            self.labels = torch.FloatTensor(labels)
            self.features_invalid = torch.FloatTensor(features_invalid)
            self.labels_invalid = torch.FloatTensor(labels_invalid)
            assert(len(self.features_invalid) == num_invalid_rows)
            logging.info(f'{num_invalid_rows} samples contained invalid values.')
            logging.info(f'{len(self.features)} samples loaded succesfully.')

    def normalize(self):
        pass

    def get_train_and_test_set(self, split_percentage):
        split_idx = round(len(self) * split_percentage)
        trainset = self.features[0:split_idx], self.labels[0:split_idx]
        testset = self.features[split_idx:len(self)], self.labels[split_idx:len(self)]
        return ZapfenDataSubSet(trainset), ZapfenDataSubSet(testset)

    def __len__(self):
        assert(len(self.features) == len(self.labels))
        return len(self.features)

    def __getitem__(self, idx):
        """Returns training samples

        Args:
            idx (int): idx of sample

        Returns:
            Tuple: features, labels
        """
        return self.features[idx], self.labels[idx]

    def _map_oeffnung(self, oeffnung):
        if oeffnung == 'o':
            return 0.0
        elif oeffnung == 'oc':
            return 0.5
        elif oeffnung == 'co':
            return 0.75
        elif oeffnung == 'c':
            return 1.0
        elif oeffnung == -1.0:  # handle magic value
            return '-1.0'
        else:
            raise ValueError('Oeffnung: Invalid value: ' + str(oeffnung))

    def _map_label(self, label_str):
        if label_str == 'P.m.ssp.m.':
            self.label_distribution['P.m.ssp.m.'] += 1
            return [1.0, 0.0, 0.0, 0.0]
        elif label_str == 'P.s.':
            self.label_distribution['P.s.'] += 1
            return [0.0, 1.0, 0.0, 0.0]
        elif label_str == 'P.s.xu.' or label_str == 'P.s.xu':
            self.label_distribution['P.s.xu.'] += 1
            return [0.0, 0.0, 1.0, 0.0]
        elif label_str == 'P.m.ssp.u.':
            self.label_distribution['P.m.ssp.u.'] += 1
            return [0.0, 0.0, 0.0, 1.0]
        else:
            raise ValueError('Label: Invalid value ' + label_str)

    def _setup_logging(self):
        f_id = datetime.now().strftime("%m.%d.%Y_%H:%M:%S_")
        filename = os.path.join('logging', 'logfiles', f_id + 'log')
        logging.basicConfig(filename=filename, filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')
        logging.basicConfig(level=logging.INFO)

    def plot_label_distribution(self):
        fig, ax = plt.subplots()
        ax.bar(self.label_distribution.keys(), self.label_distribution.values())
        plt.savefig(os.path.join('plots', 'label_distribution.png'))

    def plot_feature_distribution(self):
        analyze(xdata=self.features[0])


class ZapfenDataSubSet(ZapfenDataset):
    def __init__(self, data):
        self.features = data[0]
        self.labels = data[1]

    def __len__(self):
        assert(len(self.features) == len(self.labels))
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
