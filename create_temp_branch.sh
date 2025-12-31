#!/bin/bash
# Script to create a temporary orphan branch
# An orphan branch has no commit history and can be used for temporary work

git checkout --orphan temp_branch
git add -A
git commit -m "Initial commit"
git branch -D main
git branch -m main
