import torch
import csv
from torch.utils.data import Dataset


class ZapfenDataset(Dataset):

    def __init__(self, csv_file):
        """Loads the samples in the csv_file.

        Args:
            csv_file (str): Location of csv file.
        """
        # open csv file
        with open(csv_file, mode='r') as csv_file:
            dict_reader = csv.DictReader(csv_file, delimiter=';')

            features = []
            labels = []
            num_invalid_rows = 0

            # iterate over rows
            for row in dict_reader:
                try:

                    # filter for NA values
                    for key in dict_reader.fieldnames:
                        if key == 'Stiel_L':
                            continue
                        if row[key] == 'NA':
                            num_invalid_rows += 1
                            raise ValueError(f'"NA" value for {key}. Discarding sample!')

                    # get useful feature values
                    extracted_row = \
                        [self._map_oeffnung(row['Z_Oeffnung']), row['Z_laeng'], row['Z_breit'],
                         row['DM_Spitz'], row['DM_max'], row['DM_mit'], row['Woel_aufg'],
                         row['Woel_flach'], row['Verh_LB'], row['Asym'], row['Apo_L'],
                         row['Apo_B'], row['Apo_S'], row['Entf'], row['Verh_LABA'], row['Verh_LASA'],
                         row['Hakigkeit']]  # Stiel_L excluded, b.c. too often "NA".

                except ValueError as e:
                    print(e)
                    num_invalid_rows += 1
                    continue

                extracted_row = [float(e.replace(',', '.')) for e in extracted_row if type(e) is str]
                features.append(extracted_row)
                labels.append(self._map_label(row['Bart']))

            # convert to pytorch datatype
            self.data = torch.FloatTensor(features)
            self.labels = torch.FloatTensor(labels)
            print(f'{num_invalid_rows} samples contained invalid values.')
            print(f'{len(self.data)} samples loaded succesfully.')

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _map_oeffnung(self, oeffnung):
        if oeffnung == 'o':
            return 0.0
        elif oeffnung == 'oc':
            return 0.5
        elif oeffnung == 'co':
            return 0.75
        elif oeffnung == 'c':
            return 1.0
        else:
            raise ValueError('Oeffnung: Invalud value: ' + str(oeffnung))

    def _map_label(self, label_str):
        if label_str == 'P.m.ssp.m.':
            return [1.0, 0.0, 0.0, 0.0]
        elif label_str == 'P.s.':
            return [0.0, 1.0, 0.0, 0.0]
        elif label_str == 'P.s.xu.' or label_str == 'P.s.xu':
            return [0.0, 0.0, 1.0, 0.0]
        elif label_str == 'P.m.ssp.u.':
            return [0.0, 0.0, 0.0, 1.0]
        else:
            raise ValueError('Label: Inbalid value ' + label_str)
