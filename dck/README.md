# DCK version

This directory contains the construction-kit version of dma-3d. The original Go sources are preserved at their original paths (revision `e16103936addf61d9b3919812d6cbedcb3e58e9c`), with small asset accessors so both versions use the same embedded resources.

Run the original with `go run ./cmd/dma3d` and this version with `go run ./dck/cmd/dma3d` from the repository root.

The choreography and assets stay local; reusable rendering and effects live in `../../lib/democonstructionkit`. Second Reality retains its original ST3 music synchronization.
