# **SLURM Job Script for HPC Execution**

## **Overview**
This repository includes a **SLURM job script** (`run_script.slurm`) designed to execute scripts efficiently on **Compute Canada’s HPC clusters** (e.g., **Graham, Narval, Beluga**). The script automates **job scheduling, dependency setup, execution, and runtime tracking**, making it suitable for running our workflow.

---

## **Usage Instructions**

### **1. Preparing the Environment**
Before submitting the job, ensure:
- You have access to a **Compute Canada account**.
- You are working within a **Linux-based HPC environment** that supports **SLURM job scheduling**.
- Your scripts and dependencies are ready to run.

---

### **2. Creating the SLURM Script**
We provided `run_script.slurm` as an example script used in our experiment. Modify the script based on your specific requirements:
- **Job Name**: Update `--job-name=example_task`.
- **Email Notifications**: Replace `your_email@domain.com` with your email.
- **Account Name**: Update `--account=your_account_name` to your Compute Canada account.
- **Script Execution**: Change `python your_script.py your_parameters` to match your script and arguments.
- **Dependencies**: Modify the `pip install` command as needed.

---

### **4. Submitting the Job**
Once configured, submit the SLURM script to the cluster:
```bash
sbatch run_script.slurm
```

To check job status:
```bash
squeue --me
```

To cancel a running job:
```bash
scancel JOB_ID
```

---

### **5. Viewing Job Output**
SLURM automatically generates log files for job execution. These can be accessed using:
```bash
cat slurm-<job_id>.out
```
Replace `<job_id>` with the actual job number assigned by SLURM.

---

### **6. Expected Runtime**
- **Full-scale replication** (complete dataset): Can take several days, depending on workload and hardware availability. For optimal performance, **running on Compute Canada’s GPU-enabled nodes (P100, V100, or T4) is recommended**.

---

For further assistance, refer to Compute Canada’s **[SLURM Job Submission Guide](https://docs.computecanada.ca/wiki/Running_jobs)**.