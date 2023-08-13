# Ising Model Animator (Python)

This repository contains a Python implementation of an Ising Model animator. The Metropolis-Hastings algorithm used the following equations to generate data for pixel color array assignments:

![image](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/f20c3ca4-2b89-4660-beb5-1808f059624d)

Where the following convolution matrix corresponds to:

![image](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/359a4fc2-0ea8-4181-ae38-c2ad9349b845)

The difference in energy if the current spin is flipped is given by:

![image](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/090627cc-ab1a-4229-bc9b-be993b87c8e9)

If any of the following statements are true, then the spin is flipped :

![image](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/24e73ef1-5d4b-4ef2-b2e9-f338f88e0b95)

![image](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/2ba79f5e-d012-4159-9bc1-dad1b484a450)

Where 0.5 is a random number between 0 and 1, kb is the Boltzmann constant, T is the temperature, and dE is the difference in energy.

More information on the [Ising Model](https://en.wikipedia.org/wiki/Ising_model).

## Table of Contents

1. [Features](/README.md#features)
2. [Dependencies](/README.md#dependencies)
3. [Building and Running](/README.md#building-and-running)
4. [Example](/README.md#usage)
5. [Contributing](/README.md#contributing)
6. [License](/README.md#license)

## Features

- Prompts user with animation settings and information.
- Real-time progress bar lets the user know how many frames have been compiled during rendering.
- Parallel computing with numba for optimal performance.
-  Upon completing the animation, it lets the user know that it is saved as a GIF to the directory in which the .exe file is located.
- Graphs which show total energy and magnetization vs iterations.

## Dependencies

This project depends on the following Python libraries:

- matplotlib
- tqdm
- numpy

If these are not already installed refer to step 5 in "Building and Running".

## Building and Running

### Windows

1. Download the .zip file by pressing the green "Code" button and selecting "Download ZIP".
2. Go to the folder that the file was downloaded to and extract all components to a directory.
3. While in file explorer, click on the bar with your directory:

![image](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/827c7ffa-0e05-49ec-87f1-639f859c64ae)

4. Type cmd in this bar and press Enter.
5. If some of the dependencies are not installed, put in the following lines one at a time in your command prompt.
```
pip install matplotlib
pip install tqdm
pip install numpy
```
6. Run the following line in command prompt:
```
python -m main.py
```
7. Once completed, the generated animation and plot will be saved to this directory.

If you want to view the graphing function for the Ising Model's thermodynamic properties you can run graphCv.py for specific heat and graphM.py for magnetization in a virtual environment or in command line using the instructions above.

### Customization

If any edits want to be made to the code navigate to the main.py file to make changes and run in a virtual environment.

## Example

Animation generated by main.exe with T = 1.1, grid size = 512, iterations = 1000:

![Ising Model for 1 1 T, 100 frames, 512 grid](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/847f5816-5ad7-456d-bada-3b17dc5ada43)

Algorithm check for above animation:

![Plot for 1 1 T, 100 frames, 512 grid](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/02e3f7ce-68af-462b-a8a3-577515a0dcd2)

Specific Heat over selected Ising Model temperature range:

![Specific Heat Graph](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/a84f9cea-c61b-449f-987e-129f131c3182)

Magnetization over same temperature range and different lattice sizes:

![Magnetization as lattice size changes](https://github.com/Beniam-Kumela/ising-model-py/assets/106757076/87c08de0-ea77-4b5d-934c-d9b4ef77b084)

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request if you find a bug, have a feature request, or would like to improve the project in any way. 

For the future, I would like to implement color options for the user, faster processing with C++, and specific heat vs temperature graphs. Maybe even a real-time simulation, if possible.

## License

This project is licensed under the MIT License.
