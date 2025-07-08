import nox

@nox.session(python=["3.9"])
@nox.parametrize(
    ("numpy", "matplotlib", "scipy", "h5py", "pyyaml"),
    [("1.22", "3.5", "1.8", "3.6", "5.4")]
)
def test_minimum(session, numpy, matplotlib, scipy, h5py, pyyaml):
    session.install(f"numpy=={numpy}")
    session.install(f"matplotlib=={matplotlib}")
    session.install(f"scipy=={scipy}")
    session.install(f"h5py=={h5py}")
    session.install(f"pyyaml=={pyyaml}")

    session.install(".[test]")
    session.run("pytest")

@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def test_latest(session):
    session.install(".[test]")
    session.run("pytest")
