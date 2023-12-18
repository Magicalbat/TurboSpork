input:
    shape = (28, 28, 1);

conv_2d:
   num_filters = 16;
   kernel_size = 3;
   padding = true;

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

