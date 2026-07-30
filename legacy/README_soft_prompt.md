# Earlier soft-prompt setup note

StreamContext is the context management framework for TabPFN. Users should
first set up the relevant source code of TabPFN.

Replace the file of the same name in the
`TabPFN-main/src/tabpfn/architectures` directory with the root-level
`tabpfn_v2_6.py` from the repository to add soft-prompt support for TabPFN by
injecting it at the embedding layer.

This note documents the files that predated the current frozen-TabPFN stream
implementation. The new implementation under `src/` does not require or import
the root-level architecture replacement.
