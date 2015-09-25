set -eux

# This script pops one job off the queue and runs it

job_id=$(jobqueue lease)
job=$(jobqueue get $job_id)
eval $job
if [ "$?" -eq 0 ]; then
  jobqueue done $job_id;
else
  jobqueue exit $job_id;
fi
