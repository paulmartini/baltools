#!/bin/bash -l

# Example script to illustrate how to run the baltools balfinder
# OPTIMIZED VERSION for NERSC systems with high core counts (128 CPUs, 256 processes)

# Submit this script as: "./prepare-env.sh" instead of "sbatch prepare-env.sh"

# Prepare user env needed for Slurm batch job
# such as module load, setup runtime environment variables, or copy input files, etc.
# Basically, these are the commands you usually run ahead of the srun command 

# Load the DESI environment
source $CFS/desi/software/desi_environment.sh main

# Make sure the baltools code is in the PATH and PYTHONPATH
export PATH=$HOME/Repos/baltools/bin:$PATH
export PYTHONPATH="${PYTHONPATH}:$HOME/Repos/baltools/py"

export qsocat=/global/cfs/cdirs/desi/survey/catalogs/DA2/QSO/loa/QSO_cat_loa_main_dark_healpix_v2.fits
export altzdir=$PSCRATCH/balwork/loa-v2
export outdir=/global/cfs/cdirs/desi/users/martini/bal-catalogs/loa
export outcat=QSO_cat_loa_main_dark_healpix_v2-altbal.fits

# Optimization parameters for NERSC systems
export NPROC=256  # Number of processes (256 logical CPUs per node: 2 CPUs × 64 cores × 2 threads)
export CHUNK_SIZE=50  # Chunk size for parallel processing
export FILE_DISCOVERY_WORKERS=16  # Workers for file discovery

cat << EOF > runbal.sl
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --job-name balfinder
#SBATCH --output $outdir/job-balfinder-%j.log
#SBATCH --time=02:00:00

echo "# using this version of python"
which python

echo "# Optimization parameters:"
echo "# NPROC: $NPROC"
echo "# CHUNK_SIZE: $CHUNK_SIZE"
echo "# FILE_DISCOVERY_WORKERS: $FILE_DISCOVERY_WORKERS"

date
echo "# Step 1: Split afterburner into healpix-based redshift catalogs"
srun -N 1 splitafterburner_hp.py --qsocat $qsocat --survey main --moon dark --altzdir $altzdir --zfileroot zafter -v --nproc $NPROC --chunk-size $CHUNK_SIZE
date

echo "# Step 2: Run BAL finder on individual healpix"
srun -N 1 runbalfinder_hp.py -r loa -s main -m dark -a $altzdir -z zafter -v --nproc $NPROC --chunk-size $CHUNK_SIZE --file-discovery-workers $FILE_DISCOVERY_WORKERS --alttemp --tids
date

echo "# Step 3: Append BAL information to QSO catalog"
srun -N 1 appendbalinfo_hp.py -q $qsocat -b $altzdir -o $outdir/$outcat -m dark -s main -v --alttemp --nproc $NPROC --chunk-size $CHUNK_SIZE
date

# Other commands needed after srun, such as copy your output filies,
# should still be included in the Slurm script.
# cp <my_output_file> <target_location>/.
EOF

# Now submit the batch job
sbatch runbal.sl
