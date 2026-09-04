# RoWild

Please follow instructions from the original RoWild repository for setup and compilation. 

Our modified version of DeliBot (Monte Carlo Localization: `./mloc` in RoWild) is in the `./delibot` folder. 
This folder replaces the contents of `./rowild/gpu/src/mloc` of the RoWild benchmark. 
To compile the baseline and modified versions of DeliBot, replace `mloc` with `delibot` and use our modified `makefile.rules`. 

```
cp -r delibot rowild/gpu/src/.
cp makefile.rules rowild/gpu/.

cd rowild/gpu/src/delibot
make
```

Use `delibot/run_all.sh` to generate the execution command. Use `mloc.out` binary to establish baseline and `mloc.out.rsim` binary with the modified Vulkan-Sim simulator (`../../simulator`) to simulate execution on RoboCore. 