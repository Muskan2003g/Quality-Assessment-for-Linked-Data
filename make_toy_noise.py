import random, pathlib
p = pathlib.Path("dataset/toyKG")
ents = [line.split("\t")[0] for line in p.joinpath("entities.dict").read_text(encoding="utf-8").splitlines()]

# copy train to noise_20.txt (for demo we reuse train as "noisy")
p.joinpath("noise_20.txt").write_text(p.joinpath("train.txt").read_text(encoding="utf-8"), encoding="utf-8")

# make a small negative test file
test = [l.strip().split() for l in p.joinpath("test.txt").read_text(encoding="utf-8").splitlines()]
p.joinpath("test_negative_20.txt").write_text(
    "".join(f"{h} {r} {random.choice(ents)}\n" for h,r,t in test),
    encoding="utf-8"
)
print("toy noise files created at", p.resolve())
