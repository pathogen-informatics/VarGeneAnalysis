set -eux

# This script queues up each of the jobs use to run kmeans on each domain

for k in $(seq 3 39); do
  kp=$(($k + 1))
  for domain in DBLb DBLg DBLd DBLe CIDRb CIDRg; do
    results_folder='../../results/subsets_0,2_v_1,2'
    matrix_file="${results_folder}/input_data_subsets/matrix.${domain}.subset_012.txt"
    echo "./train_clusters_with_kmeans.py -f $k -t $kp -i 300 -s 30 -r 15 $matrix_file ${results_folder}/kmeans.subsets_012.${domain}.f${k}t${kp}i300s30r15.log" | jobqueue put

    results_folder='../../results/subsets_3,5_v_4,5'
    matrix_file="${results_folder}/input_data_subsets/matrix.${domain}.subset_345.txt"
    echo "./train_clusters_with_kmeans.py -f $k -t $kp -i 300 -s 30 -r 15 $matrix_file ${results_folder}/kmeans.subsets_345.${domain}.f${k}t${kp}i300s30r15.log" | jobqueue put
  done
done

for k in $(seq 3 39); do
  kp=$(($k + 1))
  for domain in DBLz DBLa CIDRa; do
    results_folder='../../results/subsets_0,2_v_1,2'
    matrix_file="${results_folder}/input_data_subsets/matrix.${domain}.subset_012.txt"
    echo "./train_clusters_with_kmeans.py -f $k -t $kp -i 300 -s 30 -r 15 $matrix_file ${results_folder}/kmeans.subsets_012.${domain}.f${k}t${kp}i300s30r15.log" | jobqueue put

    results_folder='../../results/subsets_3,5_v_4,5'
    matrix_file="${results_folder}/input_data_subsets/matrix.${domain}.subset_345.txt"
    echo "./train_clusters_with_kmeans.py -f $k -t $kp -i 300 -s 30 -r 15 $matrix_file ${results_folder}/kmeans.subsets_345.${domain}.f${k}t${kp}i300s30r15.log" | jobqueue put
  done
done
