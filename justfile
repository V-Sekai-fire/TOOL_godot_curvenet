default: build

# Build the Lean spec layer + the C++ algorithm header tests. Slang
# pipeline integration (Phase 3+) lands later.
build:
    cd lean && lake build
    make -C tests test

test:
    make -C tests test

lean:
    cd lean && lake build

clean:
    make -C tests clean
