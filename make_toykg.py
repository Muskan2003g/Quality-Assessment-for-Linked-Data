import pathlib

p = pathlib.Path("dataset/toyKG")
p.mkdir(parents=True, exist_ok=True)

# toy triples
p.joinpath("train.txt").write_text("\n".join(f"E{i} R0 E{i+1}" for i in range(0,180,2)), encoding="utf-8")
p.joinpath("valid.txt").write_text("E0 R0 E2\nE2 R0 E4\n", encoding="utf-8")
p.joinpath("test.txt").write_text("E4 R0 E6\nE6 R0 E8\n", encoding="utf-8")

# dict files
p.joinpath("entities.dict").write_text("\n".join(f"E{i}\t{i}" for i in range(200))+"\n", encoding="utf-8")
p.joinpath("relations.dict").write_text("\n".join(f"R{i}\t{i}" for i in range(10))+"\n", encoding="utf-8")

print("toyKG ready at", p.resolve())
