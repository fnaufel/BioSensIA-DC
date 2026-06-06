import analyze_pdb_complexes as apc

df = apc.build_complexes_frame(limit=100)
apc.write_frame(df, './scratch/2026-06-06-pairs.json')
