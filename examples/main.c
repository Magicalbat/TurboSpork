#include <stdio.h>

#include "examples.h"
#include "os.h"

int main(void) {
    ts_datetime time = ts_now_localtime();

    printf("%d\n", time.day);

    mnist_main();

    return 0;
}

