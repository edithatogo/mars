"""Spack upstream-facing package definition for the mars-earth family."""

from spack.package import PythonPackage, depends_on, version


class MarsEarth(PythonPackage):
    """Spack recipe for the `mars-earth` source distribution."""

    homepage = "https://github.com/edithatogo/mars"
    url = "https://files.pythonhosted.org/packages/source/m/mars_earth/mars_earth-{0}.tar.gz"

    version(
        "1.0.4",
        sha256="0755aa79c879e06bb83d5e2811435c20e4f81623e1ddd8451b528cd8fe6d7972",
    )

    depends_on("python@3.9:", type=("build", "run"))
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-scikit-learn", type=("build", "run"))
    depends_on("cargo", type="build")
    depends_on("rust", type="build")

    def build_args(self, spec, prefix):
        """Return build arguments for the feasibility sketch."""
        return []

    def install(self, spec, prefix):
        """Document the expected install shape for the source package."""
        super().install(spec, prefix)
