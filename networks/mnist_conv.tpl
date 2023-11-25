input:
    shape = (28, 28, 1);

conv_2d:
   num_filters = 16;
   kernel_size = (3, 3, 1);
   padding = true;
   stride_x = 1;
   stride_y = 1;
   kernels_init = std_norm;
   bias_init = zeros;

activation:
    type = relu;

pooling_2d:
    type = max;
    pool_size = (2, 2, 1);

conv_2d:
   num_filters = 32;
   kernel_size = (3, 3, 1);
   padding = true;
   stride_x = 1;
   stride_y = 1;
   kernels_init = std_norm;
   bias_init = zeros;

activation:
    type = relu;

pooling_2d:
    type = max;
    pool_size = (2, 2, 1);

flatten:

dense:
    size = 10;
    bias_init = zeros;
    weight_init = std_norm;

activation:
    type = softmax;

