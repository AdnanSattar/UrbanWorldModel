# Contributing to UrbanSim WM

Thank you for your interest in contributing to UrbanSim WM! This document provides guidelines and instructions for contributors.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Making Changes](#making-changes)
6. [Testing](#testing)
7. [Submitting Changes](#submitting-changes)
8. [Style Guidelines](#style-guidelines)
9. [Documentation](#documentation)

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Prioritize the project's goals and urban sustainability mission

---

## Getting Started

### Prerequisites

- Git
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 20+ (for local development)
- Make (optional, recommended)

### First-Time Setup

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/urbansim-wm.git
cd urbansim-wm

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/urbansim-wm.git

# Create .env file
cp .env.example .env

# Build and start services
make build
make up
```

---

## Development Setup

### Docker-Based Development (Recommended)

```bash
# Start all services with hot-reloading
make up

# View logs
make logs

# Open shell in backend
make shell-be

# Open shell in frontend
make shell-fe
```

### Local Development (Without Docker)

```bash
# Install all dependencies
make install

# Backend (Terminal 1)
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend (Terminal 2)
cd frontend
npm run dev

# Training (when needed)
cd training
python urban_world_model.py
```

Environment tips:
- Copy `.env.example` to `.env` and set relevant variables.
- For real air quality, set `WAQI_API_TOKEN`.
- Ensure `LOGS_DIR` and `PROCESSED_DATA_DIR` exist or are mounted via Docker (see `docker-compose.yml`).

---

## Project Structure

```bash
urbansim-wm/
├── backend/         # FastAPI application
├── frontend/        # Next.js application
├── training/        # Model training code
├── backend/app/etl/ # ETL sources (WAQI primary, OpenAQ fallback)
├── docs/           # Documentation (Quickstart, Implementation Summary, PRD)
└── tests/          # Test suites (to be created)
```

### Key Files

- `backend/app/main.py` - FastAPI application entry
- `backend/app/api/simulate.py` - Main simulation logic
- `backend/app/etl/waqi.py` - WAQI ETL (primary)
- `backend/app/etl/openaq.py` - OpenAQ ETL (fallback)
- `frontend/app/page.tsx` - Main UI page
- `training/urban_world_model.py` - Training harness
- `docker-compose.yml` - Service orchestration
- `Makefile` - Development commands

---

## Making Changes

### Branching Strategy

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding tests
- `chore/` - Maintenance tasks

### Commit Messages

Follow conventional commits:

```bash
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `chore:` - Maintenance

**Examples:**

```bash
feat(backend): add caching to simulation endpoint
fix(frontend): correct PM2.5 chart scaling
docs(readme): update installation instructions
test(backend): add unit tests for model wrapper
```

---

## Testing

### Backend Tests

```bash
# Install test dependencies
cd backend
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest

# With coverage
pytest --cov=app tests/
```

### Frontend Tests

```bash
# Install test dependencies
cd frontend
npm install --save-dev @testing-library/react @testing-library/jest-dom

# Run tests
npm test

# With coverage
npm test -- --coverage
```

### Integration Tests

```bash
# TODO: Add integration test suite
# docker-compose -f docker-compose.test.yml up
```

---

## Submitting Changes

### Pull Request Process

1. **Update your fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new features
   - Update documentation

3. **Test thoroughly**
   ```bash
   # Run linters
   cd backend && black . && flake8
   cd frontend && npm run lint
   
   # Run tests
   make test  # TODO: implement
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Clear title describing the change
   - Detailed description of what and why
   - Link to related issues
   - Screenshots for UI changes

Labels: please apply one or more labels (e.g., `area/backend`, `area/frontend`, `data/etl`, `docs`, `good-first-issue`).

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Reviewed own changes
- [ ] Added tests for new features
 - [ ] Updated `.env.example` or docs for new env vars
 - [ ] Added screenshots/GIFs for UI changes

---

## Style Guidelines

### Python (Backend, Training, ETL)

**Formatting:**
- Use Black for formatting: `black .`
- Follow PEP 8
- Maximum line length: 88 characters

**Style:**
```python
# Good
def simulate_policy(request: SimulationRequest) -> SimulationResponse:
    """
    Generate simulated urban dynamics.
    
    Args:
        request: Simulation parameters
        
    Returns:
        Simulation results with forecasts
    """
    # Implementation
    pass

# Bad
def simulate(r):
    # No docstring, unclear parameter
    pass
```

**Imports:**
```python
# Standard library
import os
import sys

# Third-party
import torch
import numpy as np
from fastapi import FastAPI

# Local
from app.core.config import settings
from app.models.model_wrapper import ModelWrapper
```

### TypeScript/JavaScript (Frontend)

**Formatting:**
- Use Prettier (if configured)
- 2 spaces for indentation
- Semicolons optional but consistent

**Style:**
```typescript
// Good
interface PolicyConfig {
  car_free_ratio: number
  renewable_mix: number
}

const runSimulation = async (policy: PolicyConfig): Promise<SimulationResult> => {
  // Implementation
}

// Bad
function run(p: any) {
  // No types, unclear naming
}
```

**Components:**
```tsx
// Prefer functional components with TypeScript
export default function PolicyControls({ onRun, loading }: PolicyControlsProps) {
  const [carFree, setCarFree] = useState(0.1)
  
  return (
    <div className="space-y-4">
      {/* Component content */}
    </div>
  )
}
```

### Documentation

**Docstrings (Python):**
```python
def fetch_city_air_quality(
    city: str,
    parameter: str = "pm25",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Fetch air quality measurements for a specific city.
    
    Args:
        city: City name (e.g., "Lahore", "Mumbai")
        parameter: Air quality parameter to fetch
        limit: Maximum number of measurements
        
    Returns:
        Dictionary containing API response with measurements
        
    Raises:
        requests.exceptions.RequestException: If API call fails
        
    Example:
        >>> data = fetch_city_air_quality("Lahore", "pm25", 10)
        >>> print(len(data['results']))
        10
    """
```

**Comments (TypeScript):**
```typescript
/**
 * Run simulation with given policy parameters
 * 
 * @param policy - Policy configuration object
 * @returns Promise resolving to simulation results
 */
const runSimulation = async (policy: PolicyConfig): Promise<SimulationResult> => {
  // Implementation
}
```

---

## Documentation

### When to Update Docs

- Adding new features → Update README.md
- Changing API → Update API section in README
- New configuration → Update .env.example
- Complex features → Add to docs/ directory

### Documentation Files

- `README.md` - Main documentation
- `docs/QUICKSTART.md` - Getting started guide
- `docs/IMPLEMENTATION_SUMMARY.md` - What was implemented and why
- `CONTRIBUTING.md` - This file
- `docs/` - Additional documentation
- Inline comments - For complex logic
- Docstrings - For all functions/classes

## Contribution Ideas

### High Priority

1. **Implement Real Model Training**
   - Replace stubs in `training/urban_world_model.py`
   - Implement actual RSSM, encoder, predictor
   - Add loss functions and optimization

2. **Integrate Real Data**
   - Use WAQI API (primary) with OpenAQ as fallback
   - Fetch and process historical data
   - Create training datasets

3. **Add Tests**
   - Backend unit tests
   - Frontend component tests
   - Integration tests
   - E2E tests

4. **Model Inference**
   - Load trained checkpoints
   - Replace synthetic simulation
   - Add caching

### Medium Priority

5. **Additional Features**
   - More policy parameters
   - Multi-city support
   - Historical analysis
   - Export functionality

6. **Performance**
   - API response caching
   - Model optimization
   - Database integration
   - Async task queue

7. **UI/UX Improvements**
   - Map visualizations
   - Comparison mode
   - Better mobile support (partially implemented)
   - Accessibility improvements

### Documentation & Tooling

8. **Documentation**
   - API reference
   - Architecture diagrams
   - Tutorial videos
   - Blog posts

9. **DevOps**
   - CI/CD pipeline
   - Kubernetes manifests
   - Monitoring setup
   - Load testing

---

## Getting Help

- Open an issue for bugs or questions
- Join discussions in GitHub Discussions
- Check existing issues and PRs first
- Be patient and respectful

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md (to be created)
- Mentioned in release notes
- Credited in documentation

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to UrbanSim WM! Together, we're building tools for smarter, more sustainable cities. 🌆💚**

