# rl-template

## Setting up your own repo

This repository is a github template repository.
This means that you can click the green `Use this template` button on the github page for this repo to start a new repo that is a copy of this one.
The difference between a template and a clone is that changes made in a template repository are not forwarded to the child repositories.


---
## Setting up repo
**This codebase only works with python 3.6 and above.**

Packages are stored in a `requirements.txt` file.
To install:
```
pip install -r requirements.txt
```

On machines that you do not have root access to (like compute canada machines), you will need to install in the user directory.
You can do this with:
```
pip install --user -r requirements.txt
```
Or you need to set up a virtual environment:
```
virtualenv -p python3 env
```


---
## Step-by-step example
To get set-up on compute canada:
```bash
ssh $cedar
# You should replace this github url with your repo that is a copy of the template repo
git clone https://github.com/andnp/rl-control-template.git
cd rl-control-template

# build virtual environment
virtualenv -p python3 env
. env/bin/activate

# install dependencies
pip install -r requirements.txt
```

Here is a quick guide to run an already existent experiment on compute canada.
This should run a parameter sweep over `alpha` and `epsilon` for e-greedy SARSA on MountainCar then plot the learning curve of the parameter setting that achieves the highest return averaged over 10 runs with standard error bars.
```bash
ssh $cedar
cd path/to/library
git pull # make sure you are up to date

# remove any old results that you might have lying around
# that way you don't accidentally zip them up and re-download them after the experiment
rm -rf results &

# check the cluster parameters
# make sure to balance using many parallel cpu cores
# while also being a good citizen of the resources (e.g. don't schedule 1000s of 2m jobs)
nano clusters/cedar.json

# run the experiment
python scripts/slurm.py clusters/cedar.json src/main.py ./ 10 experiments/example/*.json

# wait for a while
# then zip and download results
tar -cavf results.tar.bz2 results

# go back to your laptop
exit
scp $cedar:~/path/to/library/results.tar.bz2 ./
tar -xvf results.tar.bz2

# plot your results
python analysis/learning_curve.py experiments/example/*.json
```


---
## Dependencies
This template repo depends on a few other shared libraries to make code-splitting and sharing a little easier (for me).
The documentation and source code can be found at the following links.
* [RLGlue](https://github.com/andnp/rlglue) - my own minimal implementation of RLGlue
* [PyExpUtils](https://github.com/andnp/pyexputils) - a library containing the experiment running framework
* [PyFixedReps](https://github.com/andnp/pyfixedreps) - a few fixed representation algorithms implemented in python (e.g. tile-coding, rbfs, etc.)


---
## Organization Patterns

### Experiments
All experiments are described as completely as possible within static data files.
I choose to use `.json` files for human readability and because I am most comfortable with them.
These are stored in the `experiments` folder, usually in a subdirectory with a short name for the experiment being run (e.g. `experiments/idealH` would specify an experiment that tests the effects of using h*).

Experiment `.json` files look something like:
```jsonc
{
    "agent": "gtd2", // <-- name of your agent. these names are defined in agents/registry.py
    "problem": "randomwalk", // <-- name of the problem you're solving. these are defined in problems/registry.py
    "metaParameters": { // <-- a dictionary containing all of the meta-parameters for this particular algorithm
        "alpha": [1, 0.5, 0.25], // <-- sweep over these 3 values of alpha
        "beta": 1.0, // <-- don't sweep over beta, always use 1.0
        "use_ideal_h": true,
        "lambda": [0.0, 0.1]
    }
}
```

### Problems
I define a **problem** as a combination of:
1) environment
2) representation
3) target/behavior policies
4) number of steps
5) gamma
6) starting conditions for the agent (like in Baird's)

### results
The results are saved in a path that is defined by the experiment definition used.
The configuration for the results is specified in `config.json`.
Using the current `config.json` yields results paths that look like:
```
<base_path>/results/<experiment short name>/<agent name>/<parameter values>/errors_summary.npy
```
Where `<base_path>` is defined when you run an experiment.

### src
This is where the source code is stored.
The only `.py` files it contains are "top-level" scripts that actually run an experiment.
No utility files or shared logic at the top-level.

**agents:** contains each of the agents.
Preferably, these would be one agent per file.

**analysis:** contains shared utility code for analysing the results.
This *does not* contain scripts for analysing results, only shared logic (e.g. plotting code or results filtering).

**environments:** contains minimal implementations of just the environment dynamics.

**utils:** various utility code snippets for doing things like manipulating file paths or getting the last element of an array.
These are just reusable code chunks that have no other clear home.
I try to sort them into files that roughly name how/when they will be used (e.g. things that manipulate files paths goes in `paths.py`, things that manipulate arrays goes in `arrays.py`, etc.).

### clusters
This folder contains the job submission information that is needed to run on a cluster.
These are also `.json` files that look like:
```jsonc
{
    "account": "which compute canada account to use",
    "time": "how much time the job is expected to take",
    "nodes": "the number of cpu cores to use",
    "memPerCpu": "how much memory one parameter setting requires", // doesn't need to change
    "tasksPerNode": "how many parameter settings to run in serial on each cpu core"
}
```

Some quick terminology (that I made up and is kinda bad):
* **node**: a CPU core
* **task**: a single call to the experiment entry file (e.g. `src/main.py`). Generally only runs one parameter setting for a single run.
* **job**: a compute canada job (contains many tasks and run across multiple nodes).

The `nodes` setting determines the number of CPU cores for the job to request.
These CPU cores may not all be on the same server node and most likely will be split across several server nodes.
The job scheduling script bundled with this template repo will handle distributing jobs across multiple server nodes in the way recommended by compute canada support.

The `tasksPerNode` sets up the number of processes (calls to the experiment entry file) to be lined up per node requested.
If you request `nodes=16`, then 16 jobs will be run in **parallel**.
If you request `tasksPerNode=4`, then each node will run 4 tasks in **serial**.
In total, 64 tasks will be scheduled for one compute canada job with this configuration.
If there are 256 total tasks that need to be run, then 4 compute canada jobs will be scheduled.


---
## Running the code
There are a few layers for running the code.
The most simple layer is directly running a single experiment for a single parameter setting.
The highest layer will schedule jobs on a cluster (or on a local computer) that sweeps over all of the parameter settings.

The higher layers of running the code work by figuring out how to call the most simple layer many times, then generating a script that calls the simple layer for each parameter setting.

**Everything should be run from the root directory of the repo!**

### Directly run experiment
Let's say you want to generate a learning curve over N runs of an algorithm.
```
python src/main.py <N> <path/to/experiment.json> <parameter_setting_idx>
```
It isn't super easy to know which `parameter_setting_idx` to use.
It is more simple to make an experiment description `.json` that only contains one possible parameter permutation (i.e. has no arrays in it).
This will save the results in the results folder as specified above.

### Run parameter sweeps
If you want to run a larger experiment (i.e. a parameter sweep), you'll want to run these on a cluster (like cedar).
```
python scripts/slurm.py ./clusters/cedar.json src/main.py <path/where/results/are/saved> <num runs> <path/to/experiment.json>
```
**example:** if I want to run an experiment called `./experiments/idealH/gtd2_not.json`
```
python scripts/slurm.py ./clusters/cedar.json src/main.py ./ 100 ./experiments/idealH/gtd2_not.json
```

To run multiple experiments at once, you can specify several `.json` files.
```
python scripts/slurm.py ./clusters/cedar.json src/main.py ./ 100 ./experiments/idealH/*.json
```
or
```
python scripts/slurm.py ./clusters/cedar.json src/main.py ./ 100 ./experiments/idealH/gtd2.json ./experiments/idealH/gtd2_not.json
```

### Generate learning curves
The top-level `analysis` folder contains the scripts for generating learning curves.

```
python analysis/learning_curve.py <path/to/experiments.json>
```

**example:** One algorithm (one line)
```
python analysis/learning_curve.py ./experiments/idealH/gtd2_not.json
```

**example:** compare algorithms (multiple lines)
```
python analysis/learning_curve.py ./experiments/idealH/gtd2_not.json ./experiments/idealH/gtd2.json
```


---
## FAQs

* What are the best settings for `clusters/cedar.json`?

  As per the best practices document from compute canada, I make sure my CC jobs _always_ take at least one hour to complete.
  Because many of my **tasks** take about 5 minutes, I generally set the `tasksPerNode` parameter to ~16 to accomplish this (16*5m = 1h20m).
  I also try to make sure my jobs take no longer than 12hrs to complete (if I can help it).
  The optimal---if I can wait---is to make the jobs take just under 3hrs so that my jobs are in the highest priority queue, but put the least strain on the scheduler.
  Always leave a bit of wiggle room.

  There is a fine balance between CC job size and the number of CC jobs scheduled.
  Large CC jobs take longer to be scheduled, but a large number of small jobs put unnecessary strain on the scheduler.
  I try to limit my number of scheduled jobs to ~100 (we have a max of 2000 per person).
  To figure out how many tasks will be scheduled for an experiment, you can run:
```python
import src.experiment.ExperimentModel as Experiment

exp = Experiment.load('experiments/path/to.json')
print(exp.numPermutations())
```

* How do you get your code from your laptop to the compute canada server?

  Git is your friend.
  All of my code is always checked-in to git, and I have my experiment code cloned on my laptop and on the CC server.
  I use GitHub (or sometime bitbucket) private repos to house the code remotely.
  I make liberal use of git tags to mark checkpoints in the repo's lifespan (e.g. before I add a new contributor: `git tag before-aj-messed-things-up`, or when I submit a paper `git tag icml-2020`).
  This helps maintain my sanity when code changes and evolves over time, because now all codebase states are still accessible.

* What if one of my jobs fails or some of the tasks did not finish in time?

  One of the major advantages to the way this experiment framework is set up is that you can trivially determine exactly which results are missing after scheduling a job.
  In fact, the job scheduling script in this template repo already handles this issue by default.
  If you have results that are missing, simply run the scheduling script again with no changes and it will schedule only the missing tasks.

* I'm running the scheduling script, but it exits immediately and no jobs are scheduled?

  See the above.
  Chances are, your `results/` folder is not empty so there are no "missing results" to be scheduled.
  If you want to force the scheduling script to go forward anyways, either run `mv results results.old` or `rm -rf results/` to get rid of the results (or some other less aggressive strategy).

* Can your code use GPUs?

  Yup! Just change the bash script that is generated in `scripts/slurm.py` to request GPUs from compute canada.

* Can your code use multi-threading?

  Currently the scheduling script is not designed to handle multi-threading.
  Because my tasks tend to be relatively short (a few hours at most), and because it is generally better to have many single-threaded processes than one multi-threaded process, I have had no need to design a script to handle multi-threading.
  However, the underlying experiment framework, `PyExpUtils`, **does** have support for handling multi-threaded tasks.
  You will need to make a few modifications to `scripts/slurm.py` to change how many tasks are bundled into each job to account for using multiple threads.
  Talk to Andy if you need help!
