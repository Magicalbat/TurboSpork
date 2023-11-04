input:
    shape = (784, 1, 1);

dense:
    size = 64;

activation:
    type = leaky_relu;

dense:
    size = 10;

activation:
    type = softmax;

