# Publishing to TestPyPI

To publish the package to TestPyPI, you'll need to:

1. Create an account on TestPyPI: https://test.pypi.org/
2. Generate an API token in your account settings
3. Create a `.pypirc` file in your home directory with your credentials:

```
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-real-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token-here
```

4. Run the publish command:
```bash
twine upload --repository testpypi dist/*
```

Alternatively, you can use environment variables:
```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-your-test-token-here twine upload --repository testpypi dist/*
```