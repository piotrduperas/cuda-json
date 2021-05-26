import cudf, sys
gdf = cudf.read_json(sys.argv[1], engine='cudf', lines=True)

