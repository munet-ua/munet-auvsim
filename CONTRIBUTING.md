# Contributing to muNet-AUVsim

Thank you for your interest in contributing to **muNet-AUVsim**! We welcome contributions from the community including bug reports, feature requests, documentation improvements, and code contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Contributing Code](#contributing-code)
- [Development Workflow](#development-workflow)
  - [Fork and Clone](#1-fork-and-clone)
  - [Create a Feature Branch](#2-create-a-feature-branch)
  - [Make Your Changes](#3-make-your-changes)
  - [Test Your Changes](#4-test-your-changes)
  - [Commit Your Changes](#5-commit-your-changes)
  - [Push and Create a Pull Request](#6-push-and-create-a-pull-request)
  - [Code Review Process](#7-code-review-process)
- [Code Guidelines](#code-guidelines)
- [Commit Messages](#commit-messages)
- [Testing](#testing)
- [Documentation](#documentation)
- [Questions and Support](#questions-and-support)

## Getting Started

Before contributing, please:

1. **Read the README**: Familiarize yourself with the project structure, features, and usage
2. **Check existing issues**: See if someone has already reported the bug or requested the feature at [GitHub Issues](https://github.com/munet-ua/munet-auvsim/issues)
3. **Set up your development environment**: Follow the installation instructions in the [README](README.md)

## How to Contribute

### Reporting Bugs

Found a bug? Please open an issue on our [GitHub issue tracker](https://github.com/munet-ua/munet-auvsim/issues) with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, relevant dependencies)
- Code snippets or error messages if applicable
- Any relevant log files from `outputs/`

### Suggesting Features

Have an idea for a new feature? Open an issue with:

- A clear description of the feature
- Use cases and motivation
- Any implementation ideas you have
- Examples of how the feature would be used

### Contributing Code

We welcome code contributions! Whether you're fixing bugs, implementing features, or improving documentation, your help is appreciated.

**Please note:** We use a standard GitHub fork-and-pull-request workflow to keep contributions organized and maintainable.

## Development Workflow

We follow the standard GitHub collaborative workflow using forks and pull requests. This keeps the main repository clean while allowing anyone to contribute.

**Important**: Do not push changes directly to `main`. All changes go through the pull request review process.

### Step-by-Step Workflow

#### 1. Fork and Clone

First, create your own fork of the repository:

1. Navigate to [https://github.com/munet-ua/munet-auvsim](https://github.com/munet-ua/munet-auvsim)
2. Click the **Fork** button (top-right)
3. Clone your fork to your local machine:

```sh
# Clone your fork (replace YOUR-USERNAME with your GitHub username)
git clone https://github.com/YOUR-USERNAME/munet-auvsim.git
cd munet-auvsim

# Add the upstream repository as a remote
git remote add upstream https://github.com/munet-ua/munet-auvsim.git

# Verify remotes
git remote -v

# origin    https://github.com/YOUR-USERNAME/munet-auvsim.git (fetch)
# origin    https://github.com/YOUR-USERNAME/munet-auvsim.git (push)
# upstream  https://github.com/munet-ua/munet-auvsim.git (fetch)
# upstream  https://github.com/munet-ua/munet-auvsim.git (push)
```

#### 2. Create a Feature Branch

Always work on a feature branch, not directly on `main`:

```sh
# Update your local main branch
git checkout main
git pull upstream main

# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Examples:
# feature/add-apf-tuning
# fix/depth-controller-oscillation
# docs/update-installation-guide
```

**Branch naming conventions:**
- `feature/description` for new features
- `fix/description` for bug fixes
- `docs/description` for documentation updates
- `refactor/description` for code refactoring
- `test/description` for test additions/improvements

#### 3. Make Your Changes

Edit files, add features, fix bugs, etc. Keep your changes focused on a single topic per branch.

```sh
# Make your changes
# Edit files, test, etc.
# Check status

git status
```

#### 4. Test Your Changes

Before committing, verify everything works:

1. **Run example scripts**: Ensure `scripts/demo.py` and `scripts/example.py` still work
2. **Test your specific changes**: Run your code with various scenarios
3. **Check for regressions**: Make sure existing functionality isn't broken
4. **Test edge cases**: Consider boundary conditions and error handling

```sh
# Example: test the basic demo
python scripts/example.py

# Run your own test scenarios
python scripts/your_test_script.py
```

#### 5. Commit Your Changes

Stage and commit your changes with a clear, descriptive message:

```sh
# Stage specific files
git add path/to/modified/file.py

# Or stage all changes (use carefully)
git add .

# Commit with a descriptive message
git commit -m "Add APF repulsion parameter tuning method"
```

See [Commit Messages](#commit-messages) below for formatting guidelines.

#### 6. Push and Create a Pull Request

Push your feature branch to your fork:

```sh
# Push your branch to your fork
git push origin feature/your-feature-name
```

Then create a pull request:

1. Go to your fork on GitHub: `https://github.com/YOUR-USERNAME/munet-auvsim`
2. Click **Compare & pull request** (GitHub will prompt you after pushing)
3. Set the base repository to `munet-ua/munet-auvsim` and base branch to `main`
4. Set the compare branch to your feature branch
5. Fill in the pull request template with:
   - **Title**: Brief, descriptive summary (e.g., "Add APF repulsion parameter tuning")
   - **Description**:
     - Summary of changes
     - Related issue numbers (e.g., "Fixes #42" or "Closes #15")
     - Testing performed
     - Any breaking changes or migration notes
     - Screenshots or example output (if applicable)
6. Click **Create pull request**

**Example PR description:**

```markdown
## Summary

Implements automatic tuning for APF repulsion parameters based on vehicle density and formation requirements.

## Related Issues

Fixes \#42

## Testing Performed

- Tested with 3, 5, and 10 vehicle swarms
- Verified collision avoidance in dense formations
- Confirmed parameter stability over 1000s simulations

## Breaking Changes

None

## Checklist

- [x] Code follows project style guidelines
- [x] Docstrings added/updated
- [x] Example scripts still work
- [x] No regressions in existing functionality

```

#### 7. Code Review Process

- Maintainers will review your pull request
- Address any feedback or requested changes by pushing new commits to your branch:

```sh
# Make requested changes
git add .
git commit -m "Address review feedback: improve error handling"
git push origin feature/your-feature-name
```

- Once approved, maintainers will merge your pull request into `main`
- Your changes will be included in the next release

### Keeping Your Fork Updated

Regularly sync your fork with the upstream repository to avoid conflicts:

```sh
# Fetch upstream changes
git checkout main
git fetch upstream
git merge upstream/main

# Push updates to your fork
git push origin main

# Update your feature branch (if you have one in progress)
git checkout feature/your-feature-name
git merge main
```

Alternatively, use rebase for a cleaner history:

```sh
git checkout feature/your-feature-name
git rebase main
git push origin feature/your-feature-name --force-with-lease
```

## Code Guidelines

Please follow these guidelines to maintain code quality and consistency:

### Python Style

- Follow style used in the code base and [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Use type hints where appropriate

### Code Organization

- Add new features to appropriate existing modules
- Create new modules only when adding significant new functionality
- Keep module imports organized (standard library, third-party, local)

### Docstrings

All public functions, classes, and methods should have docstrings following the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def example_function(param1:float, param2:str='p2') -> bool:
    """Brief description of the function.
    
    More detailed explanation if needed. Can span multiple lines
    and include usage examples.
    
    Parameters
    ----------
    param1 : float
        Description of param1
    param2 : str, default='p2'
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Notes
    -----
    - Important notes or details not put in the description texts
        
    Examples
    --------
    >>> example_function(1.5, "test")
    True
    """
    pass
```

## Commit Messages

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Fix bug" not "Fixes bug")
- First line should be a brief summary (50 chars or less)
- Add detailed explanation after a blank line if needed
- Reference related issues with `#issue-number`

**Examples:**

```
Add APF repulsion parameter tuning method

Implements a new method for automatically tuning APF repulsion parameters based 
on vehicle density and formation requirements.

Fixes #42
```

```
Fix depth controller oscillation at waypoint transitions

Reduces oscillation by adjusting PID gains during waypoint approach.
```

**Commit message prefixes** (optional but recommended):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style/formatting (no functional changes)
- `refactor:` Code refactoring
- `test:` Add or modify tests
- `chore:` Maintenance tasks (dependencies, build, etc.)

## Testing

Before submitting your contribution:

1. **Test your changes**: Run your code with various scenarios
2. **Check existing functionality**: Make sure you didn't break anything
3. **Test edge cases**: Consider boundary conditions and error handling
4. **Run example scripts**: Verify `scripts/demo.py` and `scripts/example.py` still work

```sh
# Run the quick-start example
python scripts/example.py

# Run the interactive demo
python scripts/demo.py
```

If you're adding a significant feature, consider adding an example to `scripts/` demonstrating its usage.

## Documentation

Documentation is important! When contributing:

- Update docstrings for any modified functions/classes
- Update the README if you've added features or changed workflows
- Add comments for complex logic
- Update the roadmap in [README](README.md#project-status) if your contribution addresses a planned item

The project uses Sphinx for documentation. You can build the docs locally:

```sh
cd docs
pip install sphinx sphinx-copybutton sphinx-rtd-theme
make html

# View at docs/_build/html/index.html
```

Documentation is automatically built and published to [GitHub Pages](https://munet-ua.github.io/munet-auvsim/) on each merge to `main`.

## Questions and Support

Need help or have questions about contributing?

- **Bug reports**: Use the [Issue Tracker](https://github.com/munet-ua/munet-auvsim/issues)
- **Direct contact**: Reach out to the project maintainers

---

We appreciate your contributions to muNet-AUVsim! Your work helps advance underwater robotics research and multi-agent systems.
