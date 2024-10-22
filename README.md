# rl-template

# Getting started
There are typically three stages of running experiments:
1. Developing code locally
2. Running code on the cluster
3. Analyzing and plotting results

This readme is structured in a similar fashion.
I **highly** recommend going through all three steps with the example code **before** getting started on your project, unless you are an advanced user.

**Note:** things go out of date quickly. The best way to contribute to this project is to document your setup journey and ping @andnp if something goes wrong, and how you fixed it.
Or alternatively, put up a pull request in this repo updating the code or the instructions (and then ping me on slack anyways so I don't ignore you).

## Setting up your own repo

This repository is a github template repository.
This means that you can click the green `Use this template` button on the github page for this repo to start a new repo that is a copy of this one.
The difference between a template and a clone is that changes made in a template repository are not forwarded to the child repositories.
That is, once you hit the green button you will be fully divorced from me. If I go and break everything in this template repo, I will not accidentally break _your_ code.


---
## Setting up repo locally
**This codebase only works with python 3.11 and above.**

Prereqs:
* I assume you have python3.11 installed. If not, I strongly recommend setting up pyenv.
* I assume you are familiar with virtual environments.
* I assume you have rust installed. If not, I strongly recommend setting up rustup.
* You must install swig globally to build box2d-py. For example, you can use [pipx](https://pipx.pypa.io/stable/installation/), `pipx install swig`, or [Homebrew](https://brew.sh), `brew install swig`. At this time the MacPorts versions of swig may have issues building box2d-py successfully. 

Packages are stored in a `requirements.txt` file.
To install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Every time you open a new shell, _make sure you have activated the correct virtual environment_.

Now let's test your installation:
```bash
python src/main.py -e experiments/example/MountainCar/EQRC.json -i 0
```
This should start spitting out several columns of numbers. Something like:
```bash
# episode number, total steps, return, time per step, frames per second
DEBUG:exp:1 3221 -3222 1.187ms 841
DEBUG:exp:2 3458 -237 1.151ms 868
DEBUG:exp:3 3697 -239 1.111ms 899
DEBUG:exp:4 3871 -174 1.084ms 922
DEBUG:exp:5 8208 -4337 0.8237ms 1213
DEBUG:exp:6 8864 -656 0.8148ms 1227
DEBUG:exp:7 9071 -207 0.8086ms 1236
DEBUG:exp:8 9252 -181 0.8033ms 1244
DEBUG:exp:9 9401 -149 0.7999ms 1249
DEBUG:exp:10 9739 -338 0.7915ms 1263
DEBUG:exp:11 10064 -325 0.7835ms 1276
DEBUG:exp:12 10888 -824 0.7743ms 1291
DEBUG:exp:13 11131 -243 0.7693ms 1299
```
Let this run to completion, should only be a couple of minutes.
This is your primary command to do fast iteration of code. Make your changes, call this script (though probably with a different `.json` file that corresponds to your experiment).


To do experiments with continuing tasks or Optuna hyperparameter tuning, use:
```bash
python src/continuing_main.py -e experiments/continuing_example/Forager/EQRC.json -i  0
python src/optuna_tuning.py -e experiments/optuna_example/MountainCar/DQN.json -i 0
```

---
## Set up on compute canada
This is much more difficult.
**Plan to dedicate a couple of hours to setup.**

If all goes according to plan, the `scripts/setup_cc.sh` script should be enough.
But things rarely go according to plan.
```bash
ssh $cedar
cd projects/andnp
# You should replace this github url with your repo that is a copy of the template repo
git clone https://github.com/andnp/rl-control-template.git
cd rl-control-template

./scripts/setup_cc.sh
sq
```
After `setup_cc.sh` has run, you should see that you have a short job scheduled.
Go grab a coffee while you wait for this job to run.

Once you have a `venv.tar.xz` in your project directory, you are ready to continue.
```bash
# activate your **global** virtualenv
source ~/.venv/bin/activate

# schedule a **small** job
python scripts/slurm.py --clusters clusters/cedar.json --runs 5 -e experiments/example/Acrobot/*.json experiments/example/Cartpole/*.json experiments/example/MountainCar/*.json

# check that you have a few jobs scheduled (around 20)
sq
```

Once those jobs complete, you should have a `results/` folder with several `.db` files nested deeply within.
To make sure you have all of the results that you expect, just call the scheduling script again (don't forget your global virtualenv):
```bash
source ~/.venv/bin/activate
python scripts/slurm.py --clusters clusters/cedar.json --runs 5 -e experiments/example/Acrobot/*.json experiments/example/Cartpole/*.json experiments/example/MountainCar/*.json
```
This script should report that there is no more work to complete.

Finally, zip up and download your results:
```bash
tar -cavf results.tar.xz results
# go back to your computer
exit
# download the results
scp $cedar:~/projects/andnp/rl-control-template/results.tar.xz ./
# and unzip
tar -xvf results.tar.xz
```

Now you have all of the results locally and are ready to analyze them.
Results are stored in several sqlite3 databases. You can go open these yourself and use whatever plotting libraries you are comfortable with.
I also have a few utilities here for common operations (hyperparameter studies, learning curves, etc.) that load the databases into pandas `DataFrame`s.
If you are comfortable with pandas, I recommend using my code to load the results and start from there.

```bash
python experiments/example/learning_curve.py
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


---
## FAQs

* What are the best settings for `clusters/cedar.json`?

  As per the best practices document from compute canada, I make sure my CC jobs _always_ take at least one hour to complete.
  Because many of my **tasks** take about 5 minutes, I generally set the `sequential` parameter to ~16 to accomplish this (16*5m = 1h20m).
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
  I use GitHub private repos to house the code remotely.
  I make liberal use of git tags to mark checkpoints in the repo's lifespan (e.g. before I add a new contributor: `git tag before-aj-messed-things-up`, or when I submit a paper `git tag icml-2020`).
  This helps maintain my sanity when code changes and evolves over time, because now all codebase states are still accessible.

* What if one of my jobs fails or some of the tasks did not finish in time?

  One of the major advantages to the way this experiment framework is set up is that you can trivially determine exactly which results are missing after scheduling a job.
  In fact, the job scheduling script in this template repo already handles this by default.
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
