#!/bin/bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -t OrderBookBenchmark -j$(nproc) && ./OrderBookBenchmark
