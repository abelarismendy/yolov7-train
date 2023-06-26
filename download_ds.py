import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.yolo import YOLOv4DatasetExporter
import time

# Load a sample of the Open Images V7 dataset that contains 'Shoe' instances
dataset = foz.load_zoo_dataset(
    "open-images-v7", 
    split="validation", 
    label_types=["detections"], 
    classes=["Footwear"],
    max_samples=100,  # you can adjust this number based on how many samples you need
    seed=51
)

# Print summary of the dataset
print(dataset)

# Launch a FiftyOne App instance to visualize the dataset
session = fo.launch_app(dataset)

export_dir = "dataset/prueba2"

label_field = "ground_truth"  # for example

# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field=label_field,
)