Getting Started         {#gettingstarted}
===============

## Step 1: Getting a dataset

Turbospork comes with a python script to load a tensorflow dataset.
To run it, you need to have python and tensorflow_datasets installed.
Run the python script and pass in the name of the dataset you would like to load.
Use the `--norm` option to normalize the data to be between zero and one.
Use the `--out` option followed by a filename to select the output name of the file.
The default output name is `out.tst`. This script has only been tested on image datasets,
so some other datasets might not load correctly. 
Here is an example for loading the mnist dataset:

```
python dataset_gen.py mnist --norm --out mnist.tst
```

## Step 2: Setting up the C program

Before loading the data and creating a network,
you have to initialize the arena and the random number generator.
Datasets are generally large, so I am going to make the maximum size of the arena quite large.
TurboSpork comes with a function to get entorpy bytes from the os. 
I will use this function to initialize the global random number generator.
Here is the code:

```c
mga_desc desc = {
    .desired_max_size = MGA_MiB(1024),
    .desired_block_size = MGA_MiB(16),
};
// Initializing a permanent arena
mg_arena* perm_arena = mga_create(&desc);
// Setting parameters for the scratch arenas
mga_scratch_set_desc(&desc);

// Initializing the prng with a seed
u64 seeds[2] = { 0 };
ts_get_entropy(seeds, sizeof(seeds));
ts_prng_seed(seeds[0], seeds[1]);

```

## Step 3: Loading the dataset in C

You can use the `ts_tensor_list_load` to load a `*.tst` file.
From there, you can use the `ts_tensor_list_get` function to get the tensors in the list.
Here is the code for loading the data:

```c
ts_tensor_list mnist = ts_tensor_list_load(perm_arena, TS_STR8("mnist.tst"));
ts_tensor* train_imgs = ts_tensor_list_get(&mnist, TS_STR8("train_inputs"));
ts_tensor* train_labels = ts_tensor_list_get(&mnist, TS_STR8("train_labels"));
ts_tensor* test_imgs = ts_tensor_list_get(&mnist, TS_STR8("test_inputs"));
ts_tensor* test_labels = ts_tensor_list_get(&mnist, TS_STR8("test_labels"));

```

## Step 4: Creating a network layout

You can specify the layout of a neural network directly in C with the `ts_network_create` function,
but I find it easier to speicfy the network layout with a layout file (`*.tsl`).
You can also create a network layout in C and save it to a file using the `	ts_network_save_layout` funciton.
The format of the layout file is the layer name, a colon, and the layer parameters.
The first layer of every network has to be an input layer.
There are more options for layout files that I am showing here, but this is enough to get started.
Here is what the layout file looks like for a convolutional neural network:

```
input:
    shape = (28, 28, 1);

conv_2d:
   num_filters = 16;
   kernel_size = 3;

activation:
    type = relu;

pooling_2d:
    type = max;
    pool_size = (2, 2);

conv_2d:
   num_filters = 32;
   kernel_size = 3;

activation:
    type = relu;

pooling_2d:
    type = max;
    pool_size = (2, 2);

flatten:

dense:
    size = 10;

activation:
    type = softmax;

```
## Step 5: Loading the network layout

To load our network layout, use the `ts_network_load_layout` function.
You can display a summary of the network layout in the console with the `ts_network_summary` function.
Here is the code:
```c
ts_network* nn = ts_network_load_layout(perm_arena, TS_STR8("networks/mnist_conv.tsl"), true);
ts_network_summary(nn);
```

## Step 6: Training the network

To train the network, we have to create a `ts_network_train_desc` structure.
To get a full explanation of each training parameter, see the documentation for the structure.
The only unique thing I am doing for this training is the randomized transformations.
Randomly transforming the input images can help with accuracy for image recognition tasks.
To train the network, just call the `ts_network_train` function.
Here is the code:
```c
ts_network_train_desc train_desc = {
    .epochs = 32,
    .batch_size = 100,

    .num_threads = 8,

    .cost = TS_COST_CATEGORICAL_CROSS_ENTROPY,
    .optim = (ts_optimizer){
        .type = TS_OPTIMIZER_ADAM,
        .learning_rate = 0.001f,

        .adam = (ts_optimizer_adam){
            .beta1 = 0.9f,
            .beta2 = 0.999f,
            .epsilon = 1e-7f
        }
    },

    .random_transforms = true,
    .transforms = (ts_network_transforms) {
        .min_translation = -2.0f,
        .max_translation =  2.0f,

        .min_scale = 0.9f,
        .max_scale = 1.1f,

        .min_angle = -3.14159265 / 16.0f,
        .max_angle =  3.14159265 / 16.0f,
    },

    .save_interval = 4,
    .save_path = TS_STR8("training_nets/mnist_"),

    .train_inputs = train_imgs,
    .train_outputs = train_labels,

    .accuracy_test = true,
    .test_inputs = test_imgs,
    .test_outputs = test_labels
};

ts_network_train(nn, &train_desc);
```
## Step 7: Cleanup

You should delete the network and arena at the end of the program like this:
```c
ts_network_delete(nn);
mga_destroy(perm_arena);
```

To get more specifics about each step, you can look at the documentation for each header file.
