from pathlib import Path
from potdata.io.adaptor import YAMLCollectionAdaptor
from potdata.schema.datapoint import DataCollection

def merge_datasets(file1, file2, output_filename):
    # Create a YAMLCollectionAdaptor instance
    yaml_adaptor = YAMLCollectionAdaptor()

    # Read the data from the first YAML file
    data_collection1 = yaml_adaptor.read(file1)

    # Read the data from the second YAML file
    data_collection2 = yaml_adaptor.read(file2)

    # Combine the data points from both collections
    combined_data_points = data_collection1.data_points + data_collection2.data_points

    # Create a new DataCollection with the combined data points
    combined_collection = DataCollection(data_points=combined_data_points, label="Combined Training Dataset")

    # Write the combined data to a new YAML file
    yaml_adaptor.write(combined_collection, output_filename)

if __name__ == "__main__":
    # Replace these with the actual paths to your dataset files
    file1 = "../dataset_train1.yaml"
    file2 = "../dataset_train2.yaml"
    output_filename = "combined_dataset_train.yaml"

    # Merge the datasets
    merge_datasets(file1, file2, output_filename)
