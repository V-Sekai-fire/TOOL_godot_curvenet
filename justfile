default: build

build:
    scons

clean:
    scons -c
    make -C tests clean

test:
    make -C tests test
