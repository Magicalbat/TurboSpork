input:
    shape = (28, 28, 1);

conv_2d:
   num_filters = 16;
   kernel_size = (3, 3, 1);

activation:
    type = relu;

pooling_2d:
    type = max;
    pool_size = (2, 2, 1);

conv_2d:
   num_filters = 16;
   kernel_size = (3, 3, 1);

activation:
    type = relu;

pooling_2d:
    type = max;
    pool_size = (2, 2, 1);

flatten:

dense:
    size = 10;

activation:
    type = softmax;

