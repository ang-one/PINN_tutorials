In this folder you will find short examples of the usage of NN to solve PDE or ODE and how to parametrize them using data.

In the folder “vision paper” you will find two review papers about the techniques.


In the folder “Technical paper” you will find the mathematical papers of two of the main libraries for PINNs.

In the folder “”code example” you will find three simple cases of the usage of such technique. I will show in particular the library SCIML and the files project and manifest are the useful to set automatically the environment to run the codes.

We will see two different examples of integrating PDE with partial (incomplete) boundary conditions (in particular the model of the time evolution of allele frequency spectrum).\\

The script Huang_model.jl instead makes the fit of bacterial growth given a time series of optical density data. Note that you will have an example of dataset and to run this code you have to change the paths present in the script (because I am lazy).

NOTES: to start the environment, open julia copy the project folder in the chosen working directory of Julia. 
1. Using REPL or the COMMAND LINE move to the working directory.  
2. If you use the COMMAND LINE, to start a Julia session run the command:

> julia

3. To enter in the Pkg REPL  type 

>]  

4. Type the command 
> activate .

5. To install the package of the project, type
> instantiate
