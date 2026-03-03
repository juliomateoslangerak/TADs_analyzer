import os
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import json

## Define some constants
#INPUT_IMAGE_DIR = input("input directory: ")
# INPUT_IMAGE_DIR = os.path.join(".", "assays")
INPUT_IMAGE_DIR = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays"
# INPUT_IMAGE_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays"
IMAGE_FILE_EXTENSION = ".ome.tiff"


def run(assays_directory, image_files_extension, token_sep="_"):
    assay_dirs = {d for d, _, _ in os.walk(assays_directory)}

    for assay_dir in assay_dirs:
        try:
            print(assay_dir)
            with open(os.path.join(assay_dir, 'assay_config.json'), mode="r") as config_file:
                config = json.load(config_file)

            assay_id = config["assay_id"]
            column_names = config["column_names"]
            merges = config["merges"]
            fluorophores = config["fluorophores"]

            files_list = [f for f in os.listdir(assay_dir) if f.endswith(image_files_extension)]
            table = df(columns=column_names)

            for file_name in files_list:
                line = [[file_name] + file_name.split(sep=token_sep)]
                line[0][-1] = line[0][-1][:-len(image_files_extension)]
                line = df(line, columns=column_names)
                for ch in range(len(fluorophores)):
                    line[f"Label Ch{ch}"] = fluorophores[ch]
                table = pd.concat([table, line], ignore_index=True)

            for col_name, table_file in merges.items():
                merge_table = pd.read_csv(os.path.join(".", "meta_data", table_file))
                table = pd.merge(table, merge_table,
                                 how='left',
                                 on=col_name,
                                 )
            try:
                if 'NPC' in assay_dir:
                    table = table.drop(columns=["Cluster ESC Ch0"])
                    table = table.drop(columns=["Cluster ESC Ch1"])
                else:
                    table = table.drop(columns=["Cluster NPC Ch0"])
                    table = table.drop(columns=["Cluster NPC Ch1"])
            except KeyError:
                pass

            table.dropna(axis="columns", how="all", inplace=True)

            csv_header = "# header "
            for dt in list(table.dtypes):
                if dt == np.int64:
                    csv_header = csv_header + 'l,'
                elif dt == np.float64:
                    csv_header = csv_header + 'd,'
                else:
                    csv_header = csv_header + 's,'

            with open(os.path.join(assay_dir, f'{assay_id}_assays.csv'), mode='w') as csv_file:
                csv_file.write(csv_header[:-1])
                csv_file.write("\n")
                csv_file.write(table.to_csv(index=False, line_terminator='\n'))

        except FileNotFoundError:
            pass


if __name__ == "__main__":
    run(assays_directory=INPUT_IMAGE_DIR,
        image_files_extension=IMAGE_FILE_EXTENSION)
    print("done")
