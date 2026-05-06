from spack.package import PythonPackage, depends_on, version


class Pymars(PythonPackage):
    """Feasibility sketch for packaging pymars in Spack.

    This file is intentionally lane-local and does not change the runtime API.
    It records the minimum shape of a source-install recipe and the dependency
    policy needed for HPC-style feasibility checks.
    """

    homepage = "https://github.com/edithatogo/mars"
    pypi = "mars-earth/mars-earth-{version}.tar.gz"

    version("0.1.0", sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

    depends_on("python@3.10:", type=("build", "run"))
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
