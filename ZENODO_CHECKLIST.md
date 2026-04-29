# Zenodo Archival Quick Checklist

## Pre-Archival Checklist (Do This First)

- [x] **README.md** - Comprehensive documentation with installation, usage, examples
- [x] **.gitignore** - Configured to exclude large files (*.nc, *.h5, outputs/, etc.)
- [x] **LICENSE** - MIT license file in repository root
- [x] **CITATION.cff** - Citation metadata with ORCID and GitHub URL
- [x] **.zenodo.json** - Zenodo metadata (creators, keywords, subjects)
- [ ] **Git tag created** - `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] **Git tag pushed** - `git push origin v1.0.0`
- [ ] **GitHub Release created** - https://github.com/Kaushikreddym/CasCorrDiff/releases

## Zenodo Setup Steps

### Step 1: Connect GitHub to Zenodo (One-time setup)
- [ ] Go to https://zenodo.org
- [ ] Click "Sign Up" → GitHub
- [ ] Authorize Zenodo to access your GitHub account
- [ ] Go to https://zenodo.org/account/settings/github/
- [ ] Toggle "Kaushikreddym/CasCorrDiff" to ON (green)

### Step 2: Create GitHub Release
```bash
cd /beegfs/muduchuru/codes/python/CasCorrDiff

# Create and push tag
git tag -a v1.0.0 -m "Initial release: CasCorrDiff v1.0.0"
git push origin v1.0.0

# Create release on GitHub at:
# https://github.com/Kaushikreddym/CasCorrDiff/releases/new
# - Tag: v1.0.0
# - Release title: "CasCorrDiff v1.0.0"
# - Description: Brief overview and key features
```

### Step 3: Zenodo Auto-Archives
- [ ] Wait 1-5 minutes for Zenodo to auto-archive
- [ ] Check your email for DOI confirmation
- [ ] View record at: https://zenodo.org/records/RECORD_ID

### Step 4: Update Documentation with DOI
Once you have your DOI (e.g., 10.5281/zenodo.123456):

```bash
# Update CITATION.cff
repository-artifact: "https://zenodo.org/records/RECORD_ID"

# Update README with DOI badge and citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.123456.svg)](https://doi.org/10.5281/zenodo.123456)
```

## Post-Archival Steps

- [ ] Update README.md with Zenodo DOI
- [ ] Update CITATION.cff with Zenodo URL and record ID
- [ ] Add DOI badge to README
- [ ] Push updates to GitHub
- [ ] Share DOI link in project documentation

## Important Information

**Your GitHub URL:** https://github.com/Kaushikreddym/CasCorrDiff

**Your ORCID:** https://orcid.org/0000-0002-8967-7872

**Zenodo Production:** https://zenodo.org

**Zenodo Sandbox (for testing):** https://sandbox.zenodo.org

## DOI Information

- **Zenodo DOI**: Persistent identifier for your software
- **Format**: `10.5281/zenodo.XXXXXXX`
- **Usage**: Cite in publications, data management plans
- **Concept DOI**: Points to all versions (use this for general citation)
- **Version DOI**: Each release gets its own DOI

## After Archival: Full Citation

```bibtex
@software{muduchuru2026cascorrdiff,
  author = {Muduchuru, Kaushik},
  title = {CasCorrDiff: Cascading Correction Diffusion Models for Atmospheric Downscaling},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://zenodo.org/records/XXXXXXX}
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Zenodo not auto-archiving | Verify GitHub connection is ON in Zenodo settings; may need to re-authorize |
| Can't find your repo in Zenodo settings | Make sure you're logged in; try logging out/in again |
| Large files being archived | Check .gitignore is working: `git check-ignore -v filename` |
| Wrong metadata | You can edit on Zenodo before hitting "Publish" button |

## For Future Releases

When you make updates (v1.1.0, v2.0.0, etc.):

```bash
# Update version in CITATION.cff
version: "1.1.0"
date-released: YYYY-MM-DD

# Create new git tag and release
git tag -a v1.1.0 -m "Release v1.1.0: Description of changes"
git push origin v1.1.0

# Create GitHub Release (Zenodo will auto-archive)
# Each version gets its own DOI
# Concept DOI stays the same
```

---

**See ZENODO.md for detailed step-by-step instructions.**
