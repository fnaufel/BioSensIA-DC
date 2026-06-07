import analyze_pdb_complexes as apc

df = apc.build_complexes_frame()
apc.write_frame(df, '2026-06-06-pairs.json')
