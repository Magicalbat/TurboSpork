TurboSpork         {#mainpage}
==========

## Machine Learning in C

Build Instructions
------------------

### Clang

```
premake5 gmake2
make TurboSpork
```

### Visual Studio

```
premake5 vs2022
(Open Visual Studio and build TurboSpork)

```

Getting Started
---------------

TODO

Reference
---------

### Arenas

Arena are used universally throughout TurboSpork for memory managment.
**Any function that allocates memory in TurboSpork uses arenas to do so.**
TurboSpork uses `mg_arena.h` to handle arena operations.
**You should familiarize yourself with arenas and arena memory managment before using the library.**
In short, arenas are purely linear allocators.
For more detail, you can look at the [mg_arena.h github page](https://github.com/Magicalbat/mg-libraries).

