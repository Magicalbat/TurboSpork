CC = clang
CFLAGS = -m64 -std=c11 -Iturbospork/include
DEBUG_CFLAGS = -DDEBUG -g -O0 -fsanitize=address
RELEASE_CFLAGS = -DNDEBUG -O2

CFLAGS += -Wall -Wextra -pedantic -Wconversion

config ?= debug
 
ifeq ($(config), debug)
	CFLAGS += $(DEBUG_CFLAGS)
else
	CFLAGS += $(RELEASE_CFLAGS)
endif

# OS-Specific Stuff
LFLAGS = 
BIN_DIR = 
MKDIR_BIN = 
RM_BIN = 
BIN_EXT = 
LIB_EXT = 
AR = 
TS_LINK_COMMAND =

ifeq ($(OS), Windows_NT)
	BIN_DIR = bin\$(config)
	LFLAGS += -lkernel32 -lBcrypt
	MKDIR_BIN = if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	RM_BIN = rmdir /s /q bin
	BIN_EXT = .exe
	LIB_EXT = .lib
	AR = llvm-ar
	TS_LINK_COMMAND = -l$(BIN_DIR)/turbospork
else
	BIN_DIR = bin/$(config)
	LFLAGS += -lm
	MKDIR_BIN = mkdir -p $(BIN_DIR)
	RM_BIN = rm -r bin
	LIB_EXT = .a
	AR = ar
	TS_LINK_COMMAND = -L$(BIN_DIR) -l:turbospork$(LIB_EXT)
endif

all: turbospork example

turbospork:
	@$(MKDIR_BIN)
	$(CC) -c turbospork/src/turbospork.c -Iturbospork/src $(CFLAGS) $(LFLAGS) -o $(BIN_DIR)/turbospork.o
	$(AR) -rcs $(BIN_DIR)/turbospork$(LIB_EXT) $(BIN_DIR)/turbospork.o

example:
	@$(MKDIR_BIN)
	$(CC) example/main.c -Iexample $(CFLAGS) $(LFLAGS) $(TS_LINK_COMMAND) -o$(BIN_DIR)/example$(BIN_EXT)

clean:
	$(RM_BIN)

.PHONY: all turbospork example clean

