input:
    shape = (28, 28, 1);

pooling_2d:
    type = max;
    pool_size = (2, 2, 1);

flatten:

dense:
    size = 64;

activation:
    type = leaky_relu;

dense:
    size = 10;

activation:
    type = softmax;

