import torch
import os
import csv
import logging
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from torch.utils.data import Dataset


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
                extracted_row = np.array([np.float(e.replace(',', '.'))
                                          if type(e) is str
                                          else e for e in extracted_row])

                # distinguish between invalid and valid rows
                if row_is_valid:
                    features.append(extracted_row)
                    labels.append(self._map_label(row['Bart']))
                else:
                    num_invalid_rows += 1
                    features_invalid.append(extracted_row)
                    labels_invalid.append(self._map_label(row['Bart']))

            self.features = np.array(features)
            self.labels = np.array(labels)
            self.features_invalid = np.array(features_invalid)
            self.labels_invalid = np.array(labels_invalid)
            logging.info(f'{num_invalid_rows / (num_invalid_rows + len(self.features)) * 100}'
                         ' % samples contained invalid values.')

    def scale(self):
        scaler = preprocessing.StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def fix_invalid_values(self):
        """ Replaces invalid values with the respective feature mean.
        """
        logging.info(f'Fixing {len(self.features_invalid)} samples.')
        logging.info(f'{len(self.features)} present before fixing.')
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
        logging.info(f'{len(self.features)} present after fixing.')

    def get_train_and_test_set(self, split_percentage):
        split_idx = round(len(self) * split_percentage)
        return torch.utils.data.random_split(self, [split_idx, len(self) - split_idx],
                                             generator=torch.Generator().manual_seed(42))

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
        return torch.from_numpy(self.features[idx]), \
            torch.from_numpy(self.labels[idx])

    def _map_oeffnung(self, oeffnung):
        if oeffnung == 'o':
            return 0.0
        elif oeffnung == 'oc':
            return 0.25
        elif oeffnung == 'co':
            return 0.50
        elif oeffnung == 'c':
            return 0.75
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
                            level=logging.INFO)

    def plot_label_distribution(self, fname='label_distribution'):
        _, ax = plt.subplots()
        stretch_fac = sum(self.label_distribution.values())
        values = [v / stretch_fac for v in self.label_distribution.values()]
        ax.bar(self.label_distribution.keys(), values)
        plt.title('Distribution of target classes (PMF)')
        plt.savefig(os.path.join('plots', fname + '.png'))
        plt.clf()

    def plot_feature_distribution(self, fname='feature_distribution', title=''):
        ax = sns.violinplot(data=self.features, scale='width')
        desc = ['Z_Oeffnung', 'Z_laeng', 'Z_breit', 'DM_Spitz', 'DM_max',
                'DM_mit', 'Woel_aufg', 'Woel_flach', 'Verh_LB', 'Asym', 'Apo_L',
                'Apo_B', 'Apo_S', 'Entf', 'Verh_LABA', 'Verh_LASA', 'Hakigkeit', 'Stiel_L']
        ax.set_xticks(np.arange(len(desc)), labels=desc)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(os.path.join('plots', fname + '.png'))
        plt.clf()

    def fix_and_export_csv(self):
        pass  # TODO
