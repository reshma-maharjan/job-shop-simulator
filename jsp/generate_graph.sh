#!/bin/bash
dot -Tpdf cmake-build-debug-dev/operation_graph.dot -o operation_graph.pdf
dot -Tpng cmake-build-debug-dev/operation_graph.dot -o operation_graph.png
