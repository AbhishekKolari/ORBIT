#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --output=test_api_%j.out

# curl https://api.openai.com/v1/models
curl https://www.google.com