import os, nox

@nox.session(python=["3.9"])
@nox.parametrize(
    ("numpy", "matplotlib", "scipy", "h5py", "pyyaml"),
    [("1.22", "3.5", "1.8", "3.6", "5.4")]
)
def test_minimum(session, numpy, matplotlib, scipy, h5py, pyyaml):
    # Not all the minimum versions support newer versions of python
    # e.g. cannot test numpy 1.22 with python 3.12
    session.install(f"numpy=={numpy}")
    session.install(f"matplotlib=={matplotlib}")
    session.install(f"scipy=={scipy}")
    session.install(f"h5py=={h5py}")
    session.install(f"pyyaml=={pyyaml}")

    if os.getenv("GITHUB_ACTIONS") == "true":
        session.install(".[test]")
    else:
        session.install("-e", ".[test]")

    session.run("pytest", *session.posargs)

@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def test_latest(session):
    if os.getenv("GITHUB_ACTIONS") == "true":
        session.install(".[test]")
    else:
        session.install("-e", ".[test]")

    session.run("pytest", *session.posargs)
