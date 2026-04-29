# Zenodo Archival Guide for CasCorrDiff

This guide walks you through archiving the CasCorrDiff repository to Zenodo for permanent preservation and DOI generation.

## Step 1: Prepare Your Repository

### 1.1 Ensure All Required Files Exist
- ✅ README.md (comprehensive project documentation)
- ✅ LICENSE (MIT)
- ✅ CITATION.cff (citation metadata)
- ✅ .gitignore (excludes large files from git)
- ✅ requirements.txt (dependencies)

### 1.2 Clean Up Large Files

Before uploading, ensure large files are excluded:

```bash
# Verify .gitignore includes:
# - *.nc (NetCDF data files)
# - *.h5 (HDF5 model checkpoints)
# - station_extracts/
# - output/
# - wandb/
# - tensorboard/

git status  # Verify no large files are staged
```

### 1.3 Create a GitHub Release

```bash
# Create a git tag for your release
git tag -a v1.0.0 -m "Release v1.0.0: Initial Zenodo archival"
git push origin v1.0.0
```

Then create a GitHub Release at: https://github.com/Kaushikreddym/CasCorrDiff/releases
- Tag: v1.0.0
- Release title: "CasCorrDiff v1.0.0"
- Description: Include a brief overview and key features

## Step 2: Connect GitHub to Zenodo

### 2.1 Sign Into Zenodo
1. Go to https://zenodo.org (or https://sandbox.zenodo.org for testing)
2. Click "Sign Up" → Select "GitHub" → Authorize Zenodo access

### 2.2 Flip the Switch for Your Repository
1. Go to https://zenodo.org/account/settings/github/
2. Find "Kaushikreddym/CasCorrDiff" in the list
3. Click the toggle to "On" (green)
4. Zenodo will now automatically archive releases as they're created

## Step 3: Update Metadata (Optional but Recommended)

### 3.1 Create .zenodo.json

Create this file in your repository root:

```json
{
  "creators": [
    {
      "name": "Muduchuru, Kaushik",
      "affiliation": "Your Institution",
      "orcid": "0000-0002-8967-7872"
    }
  ],
  "description": "A comprehensive framework for generative downscaling and climate data correction using diffusion models. CasCorrDiff combines deterministic regression and stochastic diffusion approaches for km-scale weather prediction and extreme event analysis.",
  "keywords": [
    "downscaling",
    "diffusion models",
    "weather prediction",
    "climate modeling",
    "deep learning",
    "atmospheric science"
  ],
  "license": "MIT",
  "related_identifiers": [
    {
      "relation": "references",
      "identifier": "2309.15214",
      "resource_type": "publication",
      "scheme": "arxiv"
    }
  ],
  "access_right": "open"
}
```

Commit this to your repo:
```bash
git add .zenodo.json
git commit -m "Add Zenodo metadata"
```

### 3.2 Update CITATION.cff with Zenodo URL

After Zenodo creates a record, update:

```yaml
repository-artifact: "https://zenodo.org/records/YOUR_RECORD_ID"
```

## Step 4: Trigger Archival

### 4.1 Via GitHub Release (Recommended)

When GitHub-Zenodo connection is active:

1. Create a new Release on GitHub (already done above)
2. Zenodo automatically fetches and archives it within seconds
3. You'll receive a confirmation email with the DOI

### 4.2 Manual Upload (Alternative)

If GitHub connection fails:

1. Go to https://zenodo.org/deposit/new
2. Select "GitHub"
3. Choose "Kaushikreddym/CasCorrDiff" from the list
4. Select the v1.0.0 release tag
5. Click "Import"
6. Review/edit metadata
7. Click "Publish"

## Step 5: Verify Archival

### 5.1 Check Zenodo Record

After archival, you'll receive:
- **DOI**: `10.5281/zenodo.XXXXXXX` (persistent identifier)
- **Record URL**: `https://zenodo.org/records/XXXXXXX`
- **Citation**: Auto-generated in multiple formats

### 5.2 Update Your Documentation

Update README.md with:

```markdown
## Citation

```bibtex
@software{muduchuru2026cascorrdiff,
  author = {Muduchuru, Kaushik},
  title = {CasCorrDiff: Cascading Correction Diffusion Models for Atmospheric Downscaling},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://zenodo.org/records/XXXXXXX}
}
```

### 5.3 Use the DOI Badge

Add to your README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

## Step 6: Maintenance & Updates

### For Subsequent Releases

1. Make your code changes
2. Update version in `CITATION.cff` and setup.py (if present)
3. Create a new git tag: `git tag v1.1.0`
4. Push: `git push origin v1.1.0`
5. Create GitHub Release for v1.1.0
6. Zenodo auto-archives with new DOI
7. Each version gets its own DOI, but the **Concept DOI** (10.5281/zenodo.XXXXXXX) points to all versions

## Zenodo Sandbox (Testing)

To test before production:

1. Go to https://sandbox.zenodo.org
2. Repeat steps 1-4 with sandbox account
3. Verify everything works
4. Then do actual release on production Zenodo

## Best Practices

✅ **Do:**
- Keep .gitignore updated to exclude large data files
- Include comprehensive README
- Provide CITATION.cff for proper attribution
- Use semantic versioning (v1.0.0, v1.1.0, etc.)
- Add release notes explaining changes

❌ **Don't:**
- Commit large data files (*.nc, *.h5) - these inflate repository size
- Forget to make releases public on Zenodo
- Leave repository private on GitHub
- Mix production code with test/debug files

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Zenodo doesn't auto-archive GitHub releases | Check GitHub-Zenodo connection in Settings → Apps; may need to re-authorize |
| Large files being archived | Update .gitignore and force-push (⚠️ use with caution) |
| Wrong metadata in Zenodo | Edit directly on Zenodo record page (before publication) |
| Multiple DOIs created | Normal - each version gets its own DOI; use Concept DOI for all versions |

## Additional Resources

- [Zenodo Documentation](https://zenodo.org/help/)
- [GitHub Integration Guide](https://zenodo.org/help/github-integration/)
- [CITATION.cff Specification](https://citation-file-format.github.io/)
- [DOI Basics](https://www.doi.org/the-identifier/what-is-a-doi/)

## Summary Timeline

```
Day 1:
- [ ] Finalize code and documentation
- [ ] Create git tag and GitHub release
- [ ] Connect GitHub to Zenodo

Day 2-3:
- [ ] Zenodo auto-archives (typically within minutes)
- [ ] Receive DOI via email
- [ ] Update README with DOI
- [ ] Update CITATION.cff with Zenodo URL

Done! Your project is now permanently archived and citable.
```
