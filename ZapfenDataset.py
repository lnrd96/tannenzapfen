import torch
import os
import csv
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset
from datetime import datetime


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
                extracted_row = [float(e.replace(',', '.')) if type(e) is str else e for e in extracted_row]

                # distinguish between invalid and valid rows
                if row_is_valid:
                    features.append(extracted_row)
                    labels.append(self._map_label(row['Bart']))
                else:
                    num_invalid_rows += 1
                    features_invalid.append(extracted_row)
                    labels_invalid.append(self._map_label(row['Bart']))

            # convert to pytorch datatype
            self.features = (features)
            self.labels = (labels)
            self.features_invalid = (features_invalid)
            self.labels_invalid = (labels_invalid)
            logging.info(f'{num_invalid_rows / (num_invalid_rows + len(self.features)) * 100}'
                         ' % samples contained invalid values.')

    def normalize(self):
        scaler = preprocessing.MinMaxScaler()
        for i in range(len(self.features)):
            self.features[i] = scaler.fit_transform(self.features[i])

    def fix_invalid_values(self):
        """ Replaces invalid values with the respective feature mean.
        """
        # get means
        means = np.empty((self.features.shape[1]))
        for i in range(self.features.shape[1]):
            means[i] = np.mean(self.features[:, i])
        # iterate over invalid rows
        for row_f, row_l in zip(self.features_invalid, self.labels_invalid):
            missing_indexes = np.where(row_f == -1.0)
            logging.info(f'Replacing {len(missing_indexes[0])} features with mean.')
            if len(missing_indexes[0]) == len(row_f) or -1.0 in row_l:
                # ignore empty rows and rows with unknown label
                logging.info('Skipping row.')
                continue
            # replace missing value with mean
            for idx in missing_indexes[0]:
                row_f[idx] = means[idx]
            # add fixed row to features
            np.append(self.features, row_f)
            np.append(self.labels, row_l)
        # empty invalid rows attribute
        self.features_invalid = None
        self.labels_invalid = None

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
                            format='%(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)

    def plot_label_distribution(self, fname='label_distribution'):
        _, ax = plt.subplots()
        ax.bar(self.label_distribution.keys(), self.label_distribution.values())
        plt.title('Distribution of target classes')
        plt.savefig(os.path.join('plots', fname + '.png'))

    def plot_feature_distribution(self, fname='feature_distribution', title=''):
        ax = sns.violinplot(data=self.features, scale='width')
        ant = ['Z_Oeffnung', 'Z_laeng', 'Z_breit', 'DM_Spitz', 'DM_max',
               'DM_mit', 'Woel_aufg', 'Woel_flach', 'Verh_LB', 'Asym', 'Apo_L',
               'Apo_B', 'Apo_S', 'Entf', 'Verh_LABA', 'Verh_LASA', 'Hakigkeit', 'Stiel_L']
        ax.set_xticks(np.arange(len(ant)), labels=ant)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(os.path.join('plots', fname + '.png'))


class ZapfenDataSubSet(ZapfenDataset):
    def __init__(self, data):
        self.features = data[0]
        self.labels = data[1]

    def __len__(self):
        assert(len(self.features) == len(self.labels))
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
