# How to publish this research repo on GitHub

1. Create a **public** repository on [github.com/new](https://github.com/new) with name `cellfree-mimo-gnn-robustness`. **Do not** ask GitHub to initialize it with a README or license (the package already includes both).
2. Download and extract the provided project ZIP on your Mac.
3. Open Terminal inside the extracted `cellfree-mimo-gnn-robustness` folder.
4. Run these commands, replacing **YOUR_GITHUB_USERNAME** with your actual GitHub username:

```bash
git init
git add .
git commit -m "Add reproducible cell-free MIMO GNN pilot study"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/cellfree-mimo-gnn-robustness.git
git push -u origin main
```

5. Verify that GitHub displays `README.md`, code and `paper/main.pdf`, and that the plots render in the README.
6. Your actual repository link will then be `https://github.com/YOUR_GITHUB_USERNAME/cellfree-mimo-gnn-robustness`.
7. Add that **real link** to your CV. Do not paste the example URL without replacing the username.

**Before sharing:** Read `paper/main.pdf`, try running `pytest` and `experiment.py` on your computer, and confirm that you can explain the SE equation, per-AP softmax, iterative baseline and why FGSM is limited to *reported channel features*. This is a pilot study, not a peer-reviewed paper or a replication of existing IEEE results.

**Note:** This repo is prepared locally and **has not yet been published to GitHub**. Pushing requires authentication to your own GitHub account, and isn't done by this package.
