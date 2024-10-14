#!/bin/bash
dot -Tpdf cmake-build-debug/operation_graph.dot -o operation_graph.pdf
dot -Tpng cmake-build-debug/operation_graph.dot -o operation_graph.png
