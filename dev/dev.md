# Development

## Contributing
We welcome contributions from the community!  
- Please open an issue before starting major work to avoid duplication.  
- Submit all changes through pull requests (see branch rules below).  
- Ensure your code passes CI checks (tests and versioning are verified automatically).
- Document your changes in the root `CHANGELOG.md` file under the `## [Latest]` heading.


## Documentation
Documentation is built automatically during the release process.  

- All notebooks in the main `notebooks/` directory are copied into `docs/examples/`.  
- The `CHANGELOG.md` file from the repository root is copied into `docs/`. 
- Do **not** manually copy notebooks or changelogs into the `docs` folder.  
- The placeholders `__VERSION__` and `__DATE__` in `index.md` are automatically replaced with the current package version and release date at build time.
- Always add an entry to `CHANGELOG.md` under the `## [Latest]` heading when making changes.
- The `## [Latest]` heading will automatically be replaced with the correction version and data.
- 
## Development Branches

### Staging Branch
- All external contributions must be made through pull requests into `staging`.  
- Contributors should fork the repository, create a branch in their fork, and then open a PR.  
- Continuous Integration (CI) automatically checks that:  
  - All tests pass successfully.  
  - The version number has been increased from the latest stable release.  
- Only PRs that pass these checks will be merged. 
- Once merged, the pre-release PyPI package and the documentation are updated automatically. 

### Master Branch
- Accepts pull requests **only from `staging`**.  
- Pull requests can be approved **only by administrators**.
- Once merged, the `## [Latest]` heading in `changelog.md` will be automatically updated in both the `master` and `staging` branches
- Once merged, the main PyPI package and the documentation are updated automatically. 
- Version tags are created automatically from based on `VERSION.txt`.

### Personal Development Branches
- Maintainers may create personal development branches directly in the main repository.  
- These branches are for experimental work and are **not** part of the official release flow.  
- All changes intended for release must still go through `staging` before reaching `master`. 