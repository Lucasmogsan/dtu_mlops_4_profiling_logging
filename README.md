# Profiling
Following [These tutorials](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/profiling/)
See [documentation for python profilers](https://docs.python.org/3/library/profile.html)


## Cprofile
```bash
python -m cProfile -o <output_file> -s <sort_order> myscript.py
```

### From terminal
For sorting (only if not save to file) use ```-s```
```bash
python -m cProfile -s <sort_order> myscript.py
python -m cProfile -s cumulative vae_mnist_working.py 
```

### From output file
For saving to output file use ```-o```
```bash
python -m cProfile -o <output_file> myscript.py
```

**Visualize: Using ```pstats```** 
```bash
python -m cProfile -o profiling_output.txt vae_mnist_working.py 
```
To show the file:
- Use the ```pstats``` - e.g. from a script like 


**Visualize: Using ```snakeviz```**   
```bash
python -m cProfile -o profiling_output.prof vae_mnist_working.py
```
To show the file:
```bash
snakeviz profiling_output.prof
```


## Pytorch profiling
Using [Tensorboard profiler extension](https://pytorch.org/docs/stable/profiler.html).

After profiling to tensorboard (see ```pytorch_profiling_example.py```) a file with the ```.pt.trace.json``` is produced in the ```log/resnet...``` folder.

1. Run from terminal:
```bash
tensorboard --logdir=./log
```
2. Open tensorboard profiler page (localhost given in the terminal output)

3. Compare runs (e.g. run with both model ```model = models.resnet18()``` and ```model = models.resnet34()``` in the diff)





# Logging
Following [These tutorials](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/logging/)

## Application logging - using [logging](https://docs.python.org/3/library/logging.html)



## Experiment logging - using [wandb](https://wandb.ai/site)