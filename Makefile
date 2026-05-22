# =============================================================================
#  Makefile -- GA + GPU-STOMP
#
#  Targets:
#    make            -- release build  (-O3)
#    make debug      -- debug build    (-G -g)
#    make profile    -- Nsight build   (-lineinfo)
#    make clean      -- remove build artefacts
#
#  Adjust SM_ARCH for your GPU:
#    RTX 20xx / T4       --> sm_75
#    RTX 30xx / A100     --> sm_86   (default)
#    RTX 40xx / H100     --> sm_89
# =============================================================================

NVCC      := nvcc
SM_ARCH   := sm_86

BASE_FLAGS := -std=c++17 \
              -arch=$(SM_ARCH) \
              -Iinclude \
              --expt-relaxed-constexpr \
              --generate-line-info

REL_FLAGS  := -O3 -DNDEBUG
DBG_FLAGS  := -G -g -O0
PROF_FLAGS := -O3 -DNDEBUG -lineinfo -Xcompiler -pg

LDFLAGS    := -lm

SRC_DIR    := src
OBJ_DIR    := obj
BIN        := ga_stomp

SRCS := $(SRC_DIR)/main.cu    \
        $(SRC_DIR)/stomp.cu   \
        $(SRC_DIR)/utils.cu   \
        $(SRC_DIR)/fitness.cu \
        $(SRC_DIR)/ga.cu      \
        $(SRC_DIR)/io.cu      \
        $(SRC_DIR)/timer.cu   \
        $(SRC_DIR)/config.cu  \
        $(SRC_DIR)/benchmark.cu

OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

.PHONY: all debug profile clean

all: NVCC_FLAGS := $(BASE_FLAGS) $(REL_FLAGS)
all: $(BIN)

debug: NVCC_FLAGS := $(BASE_FLAGS) $(DBG_FLAGS)
debug: $(BIN)

profile: NVCC_FLAGS := $(BASE_FLAGS) $(PROF_FLAGS)
profile: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo "---------------------------------------------------"
	@echo "  Built: $@"
	@echo "  Run:   ./$@ [config.ini]"
	@echo "  Run:   ./$@ [n] [true_m] [pop] [gens]"
	@echo "  Run:   ./$@ --help"
	@echo "---------------------------------------------------"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN)
	@echo "Cleaned."
