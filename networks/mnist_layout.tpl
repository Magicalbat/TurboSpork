input:
    shape = (784, 1, 1);

dense:
    size = 64;

activation:
    type = leaky_relu;

dropout:
    keep_rate = 0.8;

dense:
    size = 10;

activation:
    type = softmax;

