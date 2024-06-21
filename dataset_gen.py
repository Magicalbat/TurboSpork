import tensorflow_datasets as tfds
import numpy as np
import math
import sys

# TODO: make --help output

def main():
    if len(sys.argv) == 1:
        print("Must provide a name of a dataset to load. See tensorflow-datasets for more")
        return

    dataset_name = sys.argv[1]

    if dataset_name not in tfds.list_builders():
        print("Invalid dataset name. See tensorflow-datasets for full list")
        return
    
    train_data, test_data = tfds.load(dataset_name, split=["train", "test"], as_supervised=True)

    train_data = list(tfds.as_numpy(train_data))
    test_data = list(tfds.as_numpy(test_data))

    train_inputs, train_labels = map(list, zip(*train_data))
    test_inputs, test_labels = map(list, zip(*test_data))

    train_inputs = np.array(train_inputs).astype("float32")
    test_inputs = np.array(test_inputs).astype("float32")

    if "--norm" in sys.argv or "--normalize" in sys.argv:
        train_inputs /= np.max(train_inputs).astype("float32")
        test_inputs /= np.max(test_inputs).astype("float32")

    # Converting to vector encoding
    max_label = max(train_labels)
    for i in range(len(train_labels)):
        train_labels[i] = np.array([1 if j == train_labels[i] else 0 for j in range(max_label + 1)])
    for i in range(len(test_labels)):
        test_labels[i] = np.array([1 if j == test_labels[i] else 0 for j in range(max_label + 1)])

    output_name = "out.tst"
    if "--out" in sys.argv:
        index = sys.argv.index("--out")
        if len(sys.argv) >= index:
            output_name = sys.argv[index + 1]

    with open(output_name, "wb") as f:
        # Header
        f.write(bytes("TS_tensors", "utf-8"))

        # Num tensors
        num_tensors = 4
        f.write(num_tensors.to_bytes(4, byteorder="little", signed=False))

        for name, data in zip(("train_inputs", "train_labels", "test_inputs", "test_labels"), (train_inputs, train_labels, test_inputs, test_labels)):
            f.write(len(name).to_bytes(8, byteorder="little", signed=False))
            f.write(bytes(name, "utf-8"))

            elem_shape = data[0].shape
            elem_size = math.prod(elem_shape)
            data_shape = (elem_size, 1, len(data))

            f.write(data_shape[0].to_bytes(4, byteorder="little", signed=False))
            f.write(data_shape[1].to_bytes(4, byteorder="little", signed=False))
            f.write(data_shape[2].to_bytes(4, byteorder="little", signed=False))

            if len(elem_shape) == 3:
                for elem in data:
                    for i in range(elem_shape[2]):
                        slice = elem[:,:,i].astype("float32")
                        slice.tofile(f)
            else:
                for elem in data:
                    elem = elem.astype("float32")
                    elem.tofile(f)

if __name__ == "__main__":
    main()

