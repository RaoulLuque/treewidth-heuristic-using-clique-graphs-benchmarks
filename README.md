# Treewidth Heuristic using Clique Graph Benchmarks

This is the repository containing the benchmarks for the [treewidth heuristic using clique graphs](https://github.com/RaoulLuque/treewidth-heuristic-using-clique-graphs), a heuristic that computes an upper bound the treewidth of a graph using the clique graph.

## Usage

The benchmarks are split into multiple packages which contain different benchmarks. The most important are the dimacs and k-tree benchmarks. The k-tree benchmark can be run via <br>

```cargo run --bin k-tree-benchmarks --release multithread``` <br>

However, this runs all benchmarks in ```TEST_SUITE``` in ```benchmark-suites/src/lib.rs```. For specific benchmarks from ```TEST_SUITE``` run <br>

```cargo run --bin k-tree-benchmarks --release multithread 1``` <br>

For the benchmark at index 1 in ```TEST_SUITE```. Also note the multithread in the command. With this optional argument the arguments are run in multiple threads which results in way faster total computation time. For time critical benchmarks this argument can be omitted.
Similarly to above, the dimacs-benchmarks can be run with <br>

```cargo run --bin dimacs-benchmarks --release multithread``` <br>

The results of the benchmarks can be found in the subdirectory ```benchmark_results``` of the respective package.
