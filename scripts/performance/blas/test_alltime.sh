# So far, this is the only test command. It should produce a document in ~/rocblas_benchmarks/example
python alltime.py -i ~/rocBLAS/build/release/clients/staging/ -I alltime_example.yaml -o /tmp/rocblas_example/ -w ~/rocblas_benchmarks/example -m EXECUTE PLOT DOCUMENT
