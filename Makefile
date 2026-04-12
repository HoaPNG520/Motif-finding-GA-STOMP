# =============================================================================
#  Makefile — GA + GPU-STOMP
#
#  Targets:
#    make          — build release binary (ga_stomp)
#    make debug    — build with -G -g for cuda-gdb
#    make profile  — build with nvtx markers for Nsight
#    make clean    — remove build artefacts
#
#  Requirements:
#    CUDA Toolkit ≥ 11.0
#    GPU with compute capability ≥ 7.5 (Turing/Ampere)
#    C++17 support (nvcc ≥ 11.0 provides this)
#
#  Adjust SM_ARCH for your GPU:
#    RTX 20xx / T4       → sm_75
#    RTX 30xx / A100     → sm_86
#    RTX 40xx / H100     → sm_89 / sm_90
# =============================================================================

NVCC        := nvcc
SM_ARCH     := sm_86          # ← change to match your GPU

# ── Compiler flags ────────────────────────────────────────────────────────────
BASE_FLAGS  := -std=c++17 \
               -arch=$(SM_ARCH) \
               -Iinclude \
               --expt-relaxed-constexpr \
               --generate-line-info      # enables source correlation in Nsight

REL_FLAGS   := -O3 -DNDEBUG
DBG_FLAGS   := -G -g -O0
PROF_FLAGS  := -O3 -DNDEBUG -lineinfo -Xcompiler -pg

# Link libraries
LDFLAGS     := -lm

# ── Source / object layout ────────────────────────────────────────────────────
SRC_DIR     := src
OBJ_DIR     := obj
BIN         := ga_stomp

SRCS        := $(wildcard $(SRC_DIR)/*.cu)
OBJS        := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

# ── Default target — release ──────────────────────────────────────────────────
.PHONY: all debug profile clean

all: NVCC_FLAGS := $(BASE_FLAGS) $(REL_FLAGS)
all: $(BIN)

debug: NVCC_FLAGS := $(BASE_FLAGS) $(DBG_FLAGS)
debug: $(BIN)

profile: NVCC_FLAGS := $(BASE_FLAGS) $(PROF_FLAGS)
profile: $(BIN)

# ── Link ──────────────────────────────────────────────────────────────────────
$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo "────────────────────────────────────────"
	@echo "  Built: $@"
	@echo "  Run:   ./$@ [n] [true_m] [pop] [gens]"
	@echo "────────────────────────────────────────"

# ── Compile each .cu ──────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(OBJ_DIR) $(BIN)
	@echo "Cleaned."
