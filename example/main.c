#include <stdio.h>
#include <stdlib.h>

#include <turbospork/turbospork.h>

int main(int argc, char** argv) {
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);

    int c = test_add(a, b);

    printf("%d\n", c);

    return 0;
}

