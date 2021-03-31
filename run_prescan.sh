eval "$(/data/u_ceichner_software/software/anaconda3/bin/conda shell.bash hook)"
conda activate scilpy

LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

shopt -s extglob

mkdir -p nii nii_process

dcm2niix dcm/*

mv dcm/!(*IMA) nii/

INPUTlow=$(ls nii/dcm_ep2d_diff_low_B_1000_to_2500_PAT2_*_*.nii)
IDXlow1=$(echo $INPUTlow | cut -d' ' -f 1 | cut -d'_' -f 11 | cut -d'.' -f 1)
IDXlow2=$(echo $INPUTlow | cut -d' ' -f 2 | cut -d'_' -f 11 | cut -d'.' -f 1)
MINIDXlow=$(( $IDXlow1 < $IDXlow2 ? $IDXlow1 : $IDXlow2 ))

INPUThigh=$(ls nii/dcm_ep2d_diff_high_B_2500_to_10000_PAT2_*_*.nii)
IDXhigh1=$(echo $INPUThigh | cut -d' ' -f 1 | cut -d'_' -f 11 | cut -d'.' -f 1)
IDXhigh2=$(echo $INPUThigh | cut -d' ' -f 2 | cut -d'_' -f 11 | cut -d'.' -f 1)
MINIDXhigh=$(( $IDXhigh1 < $IDXhigh2 ? $IDXhigh1 : $IDXhigh2 ))

mrcat -axis 3 nii/*_$MINIDXlow.nii nii/*_$MINIDXhigh.nii nii_process/diff_data.nii

paste -d ' ' nii/*_$MINIDXlow.bvec nii/*_$MINIDXhigh.bvec > nii_process/bvec
paste -d ' ' nii/*_$MINIDXlow.bval nii/*_$MINIDXhigh.bval > nii_process/bval

python3 scripts/deltaS_estimator.py \
  nii_process/diff_data.nii \
  nii_process/bval \
  nii_process/bvec

python3 scripts/deltaS_estimator_SS.py \
  nii_process/diff_data.nii \
  nii_process/bval \
  nii_process/bvec

python3 scripts/invMD_bmax.py \
  nii_process/diff_data.nii \
  nii_process/bval \
  nii_process/bvec \
  5000
