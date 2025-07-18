# Contributing to Neuromorphic SNN Toolkit

Thank you for your interest in contributing to the Neuromorphic SNN Toolkit! This document provides guidelines and information for contributors.

## 🤝 Ways to Contribute

- **Bug Reports**: Report issues and bugs
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve documentation, tutorials, and examples
- **Testing**: Add test cases and improve test coverage
- **Performance**: Optimize code performance and memory usage

## 🚀 Getting Started

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/snn-toolkit.git
   cd snn-toolkit
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e .[dev]
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   pytest tests/ -v
   
   # Run example to verify functionality
   python examples/mnist_classifier.py
   ```

## 🔧 Development Workflow

### Branch Strategy

- `main`: Stable release branch
- `develop`: Development branch for new features
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical hotfixes

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, readable code
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/ -v --cov=toolkit
   
   # Run specific test files
   pytest tests/test_core.py -v
   
   # Check code formatting
   black --check toolkit/ examples/ tests/
   flake8 toolkit/ examples/ tests/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new SNN layer type"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   # Create PR on GitHub
   ```

## 📝 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 88 characters (Black default)

### Code Organization

```python
#!/usr/bin/env python3
"""
Module docstring describing the purpose and functionality.
"""

import standard_library_modules
import third_party_modules
import local_modules

# Constants
CONSTANT_VALUE = 42

class ExampleClass:
    """Class docstring with description."""
    
    def __init__(self, param: int):
        """Initialize with type hints."""
        self.param = param
    
    def public_method(self) -> str:
        """Public method with docstring."""
        return self._private_method()
    
    def _private_method(self) -> str:
        """Private method indicated by underscore."""
        return f"Value: {self.param}"

def utility_function(input_data: list) -> dict:
    """
    Utility function with clear documentation.
    
    Args:
        input_data: Description of input parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
    """
    if not input_data:
        raise ValueError("Input data cannot be empty")
    
    return {"processed": input_data}
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type Hints**: Use type hints for function parameters and return values
- **Comments**: Add explanatory comments for complex logic
- **README Updates**: Update README.md for new features

### Testing Standards

- Write tests for all new functionality
- Use pytest framework
- Aim for >80% code coverage
- Include both unit tests and integration tests
- Test error conditions and edge cases

```python
import pytest
import torch
from toolkit.core import SNNLayer

class TestSNNLayer:
    """Test SNN layer functionality."""
    
    def test_layer_creation(self):
        """Test basic layer creation."""
        layer = SNNLayer(input_size=10, output_size=5)
        assert layer.input_size == 10
        assert layer.output_size == 5
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            SNNLayer(input_size=-1, output_size=5)
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Environment Information**:
  - Python version
  - PyTorch version
  - Operating system
  - GPU/CUDA information (if applicable)

- **Reproduction Steps**:
  - Minimal code example
  - Expected behavior
  - Actual behavior
  - Error messages/stack traces

- **Additional Context**:
  - Screenshots (if applicable)
  - Related issues
  - Possible solutions

### Feature Requests

For feature requests, please provide:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other possible approaches
- **Additional Context**: Examples, mockups, references

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow convention

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. **Automated Checks**: CI/CD runs tests and checks
2. **Code Review**: Maintainers review code quality
3. **Testing**: Manual testing of new features
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to maintain clean history

## 🔍 Code Review Guidelines

### For Reviewers

- **Be Constructive**: Provide helpful feedback
- **Be Specific**: Point to exact lines and suggest improvements
- **Consider Performance**: Look for efficiency improvements
- **Check Documentation**: Ensure adequate documentation
- **Test Coverage**: Verify sufficient test coverage

### For Contributors

- **Respond Promptly**: Address review comments quickly
- **Ask Questions**: Clarify unclear feedback
- **Update Documentation**: Keep docs in sync with code changes
- **Be Patient**: Review process ensures code quality

## 🏗️ Architecture Guidelines

### Adding New Backends

When adding support for new SNN backends:

1. **Create Backend Module**: `toolkit/backends/new_backend.py`
2. **Implement Interface**: Follow existing backend patterns
3. **Add Tests**: Comprehensive test coverage
4. **Update Documentation**: Backend comparison and setup
5. **Add Example**: Show backend-specific features

### Adding New Features

For major new features:

1. **Design Document**: Create RFC for significant changes
2. **API Design**: Follow existing patterns
3. **Backward Compatibility**: Maintain existing APIs
4. **Performance**: Consider computational efficiency
5. **Documentation**: Complete user guide

## 📚 Documentation Guidelines

### Code Documentation

- **Module Docstrings**: Describe module purpose
- **Class Docstrings**: Explain class functionality
- **Method Docstrings**: Detail parameters and returns
- **Inline Comments**: Explain complex logic

### User Documentation

- **Tutorials**: Step-by-step guides
- **API Reference**: Complete parameter documentation
- **Examples**: Working code samples
- **FAQ**: Common questions and solutions

## 🚢 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in setup.py
- [ ] Git tag created
- [ ] PyPI package uploaded
- [ ] GitHub release created

## 🤔 Questions?

- **GitHub Discussions**: [Ask questions](https://github.com/neuromorphic-ai/snn-toolkit/discussions)
- **Issues**: [Report bugs](https://github.com/neuromorphic-ai/snn-toolkit/issues)
- **Discord**: [Join our community](https://discord.gg/neuromorphic-ai)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the Neuromorphic SNN Toolkit! 🧠⚡**

*Together, we're making spiking neural networks accessible to everyone.*