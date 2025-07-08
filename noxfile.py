import nox

@nox.session(python=["3.9", "3.12"])
@nox.parametrize(
    ("numpy", "matplotlib", "scipy", "h5py", "pyyaml"),
    [
        ("1.22", "3.5", "1.8", "3.6", "5.4"),   # minimum versions
        ("latest", "latest", "latest", "latest", "latest"),  # latest versions
    ]
)
def test_matrix(session, numpy, matplotlib, scipy, h5py, pyyaml):
    packages = dict(
        numpy=numpy,
        matplotlib=matplotlib,
        scipy=scipy,
        h5py=h5py,
        pyyaml=pyyaml,
    )
    for package, version in packages.items():
        session.install(package if version == "latest" else f"{package}=={version}")

    session.install(".[test]")
    session.run("pytest")